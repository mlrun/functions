import os
from typing import Literal, TypedDict, Annotated, Sequence, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from operator import add

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers import Run
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool, BaseTool

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_mlrun import (
    mlrun_monitoring,
    MLRunTracer,
    MLRunTracerSettings,
    MLRunTracerClientSettings,
    MLRunTracerMonitorSettings,
    mlrun_monitoring_env_var,
    LangChainMonitoringApp,
)


def _check_openai_credentials() -> bool:
    """
    Check if OpenAI API key is set in environment variables.

    :return: True if OPENAI_API_KEY is set, False otherwise.
    """
    return "OPENAI_API_KEY" in os.environ


# Import ChatOpenAI only if OpenAI credentials are available (meaning `langchain-openai` must be installed).
if _check_openai_credentials():
    from langchain_openai import ChatOpenAI


class _ToolEnabledFakeModel(GenericFakeChatModel):
    """
    A fake chat model that supports tool binding for running agent tracing tests.
    """

    def bind_tools(
        self,
        tools: Sequence[
            dict[str, Any] | type | Callable | BaseTool  # noqa: UP006
        ],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


#: Tag value for testing tag filtering.
_dummy_tag = "dummy_tag"


def _run_simple_chain() -> str:
    """
    Run a simple LangChain chain that gets a fact about a topic.
    """
    # Build a simple chain: prompt -> llm -> str output parser
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        tags=[_dummy_tag]
    ) if _check_openai_credentials() else (
        FakeListChatModel(
            responses=[
                "MLRun is an open-source orchestrator for machine learning pipelines."
            ],
            tags=[_dummy_tag]
        )
    )
    prompt = ChatPromptTemplate.from_template("Tell me a short fact about {topic}")
    chain = prompt | llm | StrOutputParser()

    # Run the chain:
    response = chain.invoke({"topic": "MLRun"})
    return response


def _run_simple_agent():
    """
    Run a simple LangChain agent that uses two tools to get weather and stock price.
    """
    # Define the tools:
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a specific city."""
        return f"The weather in {city} is 22°C and sunny."

    @tool
    def get_stock_price(symbol: str) -> str:
        """Get the current stock price for a symbol."""
        return f"The stock price for {symbol} is $150.25."

    # Define the model:
    model = ChatOpenAI(
        model="gpt-4o-mini",
        tags=[_dummy_tag]
    ) if _check_openai_credentials() else (
        _ToolEnabledFakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "args": {"city": "London"}, "id": "call_abc123"},
                            {"name": "get_stock_price", "args": {"symbol": "AAPL"}, "id": "call_def456"}
                        ]
                    ),
                    AIMessage(content="The weather in London is 22°C and AAPL is trading at $150.25.")
                ]
            ),
            tags=[_dummy_tag]
        )
    )

    # Create the agent:
    agent = create_agent(
        model=model,
        tools=[get_weather, get_stock_price],
        system_prompt="You are a helpful assistant with access to tools."
    )

    # Run the agent:
    return agent.invoke({"messages": ["What is the weather in London and the stock price of AAPL?"]})


def _run_langgraph_graph():
    """
    Run a LangGraph agent that uses reflection to correct its answer.
    """

    # Define the graph state:
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add]
        attempts: int

    # Define the model:
    model = ChatOpenAI(model="gpt-4o-mini") if _check_openai_credentials() else (
        _ToolEnabledFakeModel(
            messages=iter(
                [
                    AIMessage(content="There are 2 'r's in Strawberry."),  # Mocking the failure
                    AIMessage(content="I stand corrected. S-t-r-a-w-b-e-r-r-y. There are 3 'r's."),  # Mocking the fix
                ]
            )
        )
    )

    # Define the graph nodes and router:
    def call_model(state: AgentState):
        response = model.invoke(state["messages"])
        return {"messages": [response], "attempts": state["attempts"] + 1}

    def reflect_node(state: AgentState):
        prompt = "Wait, count the 'r's again slowly, letter by letter. Are you sure?"
        return {"messages": [HumanMessage(content=prompt)]}

    def router(state: AgentState) -> Literal["reflect", END]:
        # Make sure there are 2 attempts at least for an answer:
        if state["attempts"] == 1:
            return "reflect"
        return END

    # Build the graph:
    builder = StateGraph(AgentState)
    builder.add_node("model", call_model)
    tagged_reflect_node = RunnableLambda(reflect_node).with_config(tags=[_dummy_tag])
    builder.add_node("reflect", tagged_reflect_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", router)
    builder.add_edge("reflect", "model")
    graph = builder.compile()

    # Run the graph:
    return graph.invoke({"messages": [HumanMessage(content="How many 'r's in Strawberry?")], "attempts": 0})


#: List of example functions to run in tests along the full (split-run enabled) expected monitor events.
_run_suites: list[tuple[Callable, int]] = [
    (_run_simple_chain, 4),
    (_run_simple_agent, 9),
    (_run_langgraph_graph, 9),
]


#: Dummy environment variables for testing.
_dummy_environment_variables = {
    "MLRUN_TRACER_CLIENT_V3IO_STREAM_PATH": "dummy_stream_path",
    "MLRUN_TRACER_CLIENT_V3IO_CONTAINER": "dummy_container",
    "MLRUN_TRACER_CLIENT_MODEL_ENDPOINT_NAME": "dummy_model_name",
    "MLRUN_TRACER_CLIENT_MODEL_ENDPOINT_UID": "dummy_model_endpoint_uid",
    "MLRUN_TRACER_CLIENT_SERVING_FUNCTION": "dummy_serving_function",
    "MLRUN_TRACER_MONITOR_DEBUG": "true",
    "MLRUN_TRACER_MONITOR_DEBUG_TARGET_LIST": "true",
    "MLRUN_TRACER_MONITOR_SPLIT_RUNS": "true",
}


@pytest.fixture()
def auto_mode_settings(monkeypatch):
    """
    Sets the environment variables to enable mlrun monitoring in 'auto' mode.
    """
    # Set environment variables for the duration of the test:
    monkeypatch.setenv(mlrun_monitoring_env_var, "1")
    for key, value in _dummy_environment_variables.items():
        monkeypatch.setenv(key, value)

    # Reset the singleton tracer to ensure fresh initialization:
    MLRunTracer._singleton_tracer = None
    MLRunTracer._initialized = False

    yield

    # Reset the singleton tracer after the test:
    MLRunTracer._singleton_tracer = None
    MLRunTracer._initialized = False


@pytest.fixture
def manual_mode_settings():
    """
    Sets the mandatory client settings and debug flag for the tests.
    """
    settings = MLRunTracerSettings(
        client=MLRunTracerClientSettings(
            v3io_stream_path="dummy_stream_path",
            v3io_container="dummy_container",
            model_endpoint_name="dummy_model_name",
            model_endpoint_uid="dummy_model_endpoint_uid",
            serving_function="dummy_serving_function",
        ),
        monitor=MLRunTracerMonitorSettings(
            debug=True,
            debug_target_list=[],
            split_runs=True,  # Easier to test with split runs (filters can filter per run instead of inner events)
        ),
    )

    yield settings


def test_settings_init_via_env_vars(monkeypatch):
    """
    Test that settings are correctly initialized from environment variables.
    """
    #: First, ensure that without env vars, validation fails due to missing required fields:
    with pytest.raises(ValidationError):
        MLRunTracerSettings()

    # Now, set the environment variables for the client settings and debug flag:
    for key, value in _dummy_environment_variables.items():
        monkeypatch.setenv(key, value)

    # Ensure that settings are now correctly initialized from env vars:
    settings = MLRunTracerSettings()
    assert settings.client.v3io_stream_path == "dummy_stream_path"
    assert settings.client.v3io_container == "dummy_container"
    assert settings.client.model_endpoint_name == "dummy_model_name"
    assert settings.client.model_endpoint_uid == "dummy_model_endpoint_uid"
    assert settings.client.serving_function == "dummy_serving_function"
    assert settings.monitor.debug is True


@pytest.mark.parametrize(
    "test_suite", [
        # Valid case: only v3io settings provided
        (
            {
                "v3io_stream_path": "dummy_stream_path",
                "v3io_container": "dummy_container",
                "model_endpoint_name": "dummy_model_name",
                "model_endpoint_uid": "dummy_model_endpoint_uid",
                "serving_function": "dummy_serving_function",
            },
            True,
        ),
        # Invalid case: partial v3io settings provided
        (
            {
                "v3io_stream_path": "dummy_stream_path",
                "model_endpoint_name": "dummy_model_name",
                "model_endpoint_uid": "dummy_model_endpoint_uid",
                "serving_function": "dummy_serving_function",
            },
            False,
        ),
        # Valid case: only kafka settings provided
        (
            {
                "kafka_broker": "dummy_bootstrap_servers",
                "kafka_topic": "dummy_topic",
                # TODO: Add more mandatory kafka settings
                "model_endpoint_name": "dummy_model_name",
                "model_endpoint_uid": "dummy_model_endpoint_uid",
                "serving_function": "dummy_serving_function",
            },
            True,
        ),
        # Invalid case: partial kafka settings provided
        (
            {
                "kafka_broker": "dummy_bootstrap_servers",
                "model_endpoint_name": "dummy_model_name",
                "model_endpoint_uid": "dummy_model_endpoint_uid",
                "serving_function": "dummy_serving_function",
            },
            False,
        ),
        # Invalid case: both v3io and kafka settings provided
        (
            {
                "v3io_stream_path": "dummy_stream_path",
                "v3io_container": "dummy_container",
                "kafka_broker": "dummy_bootstrap_servers",
                "kafka_topic": "dummy_topic",
                # TODO: Add more mandatory kafka settings
                "model_endpoint_name": "dummy_model_name",
                "model_endpoint_uid": "dummy_model_endpoint_uid",
                "serving_function": "dummy_serving_function",
            },
            False,
        ),
        # Invalid case: both v3io and kafka settings provided (partial)
        (
            {
                "v3io_container": "dummy_container",
                "kafka_broker": "dummy_bootstrap_servers",
                "model_endpoint_name": "dummy_model_name",
                "model_endpoint_uid": "dummy_model_endpoint_uid",
                "serving_function": "dummy_serving_function",
            },
            False,
        ),
    ]
)
def test_settings_v3io_kafka_combination(test_suite: tuple[dict[str, str], bool]):
    """
    Test that settings validation enforces mutual exclusivity between v3io and kafka configurations.

    :param test_suite: A tuple containing environment variable overrides and a flag indicating
        whether validation should pass.
    """
    settings, should_pass = test_suite

    if should_pass:
        MLRunTracerClientSettings(**settings)
    else:
        with pytest.raises(ValidationError):
            MLRunTracerClientSettings(**settings)


def test_auto_mode_singleton_thread_safety(auto_mode_settings):
    """
    Test that MLRunTracer singleton initialization is thread-safe in 'auto' mode.

    :param auto_mode_settings: Fixture to set up 'auto' mode environment and settings.
    """
    # Initialize a list to hold tracer instances created in different threads:
    tracer_instances = []

    # Function to initialize the tracer in a thread:
    def _init_tracer():
        tracer = MLRunTracer()
        return tracer

    # Use ThreadPoolExecutor to simulate concurrent tracer initialization:
    num_threads = 50
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_init_tracer) for _ in range(num_threads)]
        tracer_instances = [f.result() for f in futures]

    # Check if every single reference in the list is the exact same object:
    unique_instances = set(tracer._uid for tracer in tracer_instances)

    assert len(tracer_instances) == num_threads, "Not all threads returned a tracer instance. Test cannot proceed."
    assert len(unique_instances) == 1, (
        f"Thread-safety failure! {len(unique_instances)} different instances were created under high concurrency."
    )
    assert tracer_instances[0] is MLRunTracer(), "The global access point should return the same singleton."


def test_manual_mode_multi_instances(manual_mode_settings: MLRunTracerSettings):
    """
    Test that MLRunTracer allows multiple instances in 'manual' mode.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    """
    # Initialize a list to hold tracer instances created in different iterations:
    tracer_instances = []

    # Create multiple tracer instances:
    num_instances = 50
    for _ in range(num_instances):
        tracer = MLRunTracer(settings=manual_mode_settings)
        tracer_instances.append(tracer)

    # Check if every single reference in the list is a different object:
    unique_instances = set(tracer._uid for tracer in tracer_instances)

    assert len(tracer_instances) == num_instances, "Not all instances were created. Test cannot proceed."
    assert len(unique_instances) == num_instances, (
        f"Manual mode failure! {len(unique_instances)} unique instances were created instead of {num_instances}."
    )


@pytest.mark.parametrize("run_suites", _run_suites)
def test_auto_mode(auto_mode_settings, run_suites: tuple[Callable, int]):
    """
    Test that MLRunTracer in 'auto' mode captures debug target list after running a LangChain / LangGraph example code.

    :param auto_mode_settings: Fixture to set up 'auto' mode environment and settings.

    :param run_suites: The function to run with the expected monitored events.
    """
    run_func, expected_events = run_suites

    tracer = MLRunTracer()
    assert len(tracer.settings.monitor.debug_target_list) == 0

    print(run_func())
    assert len(tracer.settings.monitor.debug_target_list) == expected_events


@pytest.mark.parametrize("run_suites", _run_suites)
def test_manual_mode(manual_mode_settings: MLRunTracerSettings, run_suites: tuple[Callable, int]):
    """
    Test that MLRunTracer in 'auto' mode captures debug target list after running a LangChain / LangGraph example code.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events.
    """
    run_func, expected_events = run_suites

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        print(run_func())
        assert len(tracer.settings.monitor.debug_target_list) == expected_events


def test_labeling(manual_mode_settings: MLRunTracerSettings):
    """
    Test that MLRunTracer in 'auto' mode captures debug target list after running a LangChain / LangGraph example code.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    """
    for i, (run_func, expected_events) in enumerate(_run_suites):
        label = f"label_{i}"
        manual_mode_settings.monitor.label = label
        manual_mode_settings.monitor.debug_target_list.clear()
        with mlrun_monitoring(settings=manual_mode_settings) as tracer:
            print(run_func())
            assert len(tracer.settings.monitor.debug_target_list) == expected_events
            for event in tracer.settings.monitor.debug_target_list:
                assert event["label"] == label


@pytest.mark.parametrize(
    "run_suites", [
        run_suite + (filtered_events,)
        for run_suite, filtered_events in zip(_run_suites, [1, 2, 1])
    ]
)
def test_monitor_settings_tags_filter(
    manual_mode_settings: MLRunTracerSettings,
    run_suites: tuple[Callable, int, int],
):
    """
    Test the `tags_filter` setting of MLRunTracer.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events and filtered events.
    """
    run_func, expected_events, filtered_events = run_suites

    manual_mode_settings.monitor.tags_filter = [_dummy_tag]

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        print(run_func())
        assert len(tracer.settings.monitor.debug_target_list) == filtered_events
        for event in tracer.settings.monitor.debug_target_list:
            assert not set(manual_mode_settings.monitor.tags_filter).isdisjoint(event["input_data"]["input_data"]["tags"])


@pytest.mark.parametrize(
    "run_suites", [
        run_suite + (filtered_events,)
        for run_suite, filtered_events in zip(_run_suites, [1, 3, 4])
    ]
)
def test_monitor_settings_name_filter(
    manual_mode_settings: MLRunTracerSettings,
    run_suites: tuple[Callable, int, int],
):
    """
    Test the `names_filter` setting of MLRunTracer.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events and filtered events.
    """
    run_func, expected_events, filtered_events = run_suites

    manual_mode_settings.monitor.names_filter = ["StrOutputParser", "get_weather", "model", "router"]

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        print(run_func())
        assert len(tracer.settings.monitor.debug_target_list) == filtered_events
        for event in tracer.settings.monitor.debug_target_list:
            assert event["input_data"]["input_data"]["run_name"] in manual_mode_settings.monitor.names_filter


@pytest.mark.parametrize(
    "run_suites", [
        run_suite + (filtered_events,)
        for run_suite, filtered_events in zip(_run_suites, [2, 7, 9])
    ]
)
@pytest.mark.parametrize("split_runs", [True, False])
def test_monitor_settings_run_type_filter(
    manual_mode_settings: MLRunTracerSettings,
    run_suites: tuple[Callable, int, int],
    split_runs: bool
):
    """
    Test the `run_types_filter` setting of MLRunTracer. Will also test with split runs enabled and disabled - meaning
    that when disabled, if a parent run is filtered, all its child runs are also filtered by default. In the test we
    made sure that the root run is always passing the filter (hence the equal one).

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events and filtered events.
    :param split_runs: Whether to enable split runs in the monitor settings.
    """
    run_func, expected_events, filtered_events = run_suites
    filtered_events = filtered_events if split_runs else 1

    manual_mode_settings.monitor.run_types_filter = ["llm", "chain"]
    manual_mode_settings.monitor.split_runs = split_runs

    def recursive_check_run_types(run: dict):
        assert run["input_data"]["run_type"] in manual_mode_settings.monitor.run_types_filter
        if "child_runs" in run["output_data"]:
            for child_run in run["output_data"]["child_runs"]:
                recursive_check_run_types(child_run)

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        print(run_func())
        assert len(tracer.settings.monitor.debug_target_list) == filtered_events

        for event in tracer.settings.monitor.debug_target_list:
            event_run = {
                "input_data": event["input_data"]["input_data"],
                "output_data": event["output_data"]["output_data"],
            }
            recursive_check_run_types(run=event_run)

@pytest.mark.parametrize("run_suites", _run_suites)
@pytest.mark.parametrize("split_runs", [True, False])
def test_monitor_settings_full_filter(
    manual_mode_settings: MLRunTracerSettings,
    run_suites: tuple[Callable, int],
    split_runs: bool
):
    """
    Test that a complete filter (not allowing any events to pass) won't fail the tracer.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events.
    :param split_runs: Whether to enable split runs in the monitor settings.
    """
    run_func, _ = run_suites

    manual_mode_settings.monitor.run_types_filter = ["dummy_run_type"]
    manual_mode_settings.monitor.split_runs = split_runs

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        print(run_func())
        assert len(tracer.settings.monitor.debug_target_list) == 0


@pytest.mark.parametrize("run_suites", _run_suites)
@pytest.mark.parametrize("split_runs", [True, False])
@pytest.mark.parametrize("root_run_only", [True, False])
def test_monitor_settings_split_runs_and_root_run_only(
    manual_mode_settings: MLRunTracerSettings,
    run_suites: tuple[Callable, int],
    split_runs: bool,
    root_run_only: bool,
):
    """
    Test the `split_runs` setting of MLRunTracer.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events.
    :param split_runs: Whether to enable split runs in the monitor settings.
    :param root_run_only: Whether to enable `root_run_only` in the monitor settings.
    """
    run_func, expected_events = run_suites

    manual_mode_settings.monitor.split_runs = split_runs
    manual_mode_settings.monitor.root_run_only = root_run_only

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        for run_iteration in range(1, 3):
            print(run_func())
            if root_run_only:
                assert len(tracer.settings.monitor.debug_target_list) == 1 * run_iteration
                assert "child_runs" not in tracer.settings.monitor.debug_target_list[-1]["output_data"]["output_data"]
            elif split_runs:
                assert len(tracer.settings.monitor.debug_target_list) == expected_events * run_iteration
                assert "child_runs" not in tracer.settings.monitor.debug_target_list[-1]["output_data"]["output_data"]
            else:  # split_runs disabled
                assert len(tracer.settings.monitor.debug_target_list) == 1 * run_iteration
                assert len(tracer.settings.monitor.debug_target_list[-1]["output_data"]["output_data"]["child_runs"]) != 0


class _CustomRunSummarizerSettings(BaseSettings):
    """
    Settings for the custom summarizer function.
    """
    dummy_value: int = 21

    model_config = SettingsConfigDict(env_prefix="TEST_CUSTOM_SUMMARIZER_SETTINGS_")


def _custom_run_summarizer(run: Run, settings: _CustomRunSummarizerSettings = None):
    """
    A custom summarizer function for testing.

    :param run: The LangChain / LangGraph run to summarize.
    :param settings: Optional settings for the summarizer.
    """
    inputs = {
        "run_id": run.id,
        "input": run.inputs,
        "from_settings": settings.dummy_value if settings else 0,
    }

    def count_llm_calls(r: Run) -> int:
        if not r.child_runs:
            return 1 if r.run_type == "llm" else 0
        return sum(count_llm_calls(child) for child in r.child_runs)

    def count_tool_calls(r: Run) -> int:
        if not r.child_runs:
            return 1 if r.run_type == "tool" else 0
        return sum(count_tool_calls(child) for child in r.child_runs)

    outputs = {
        "llm_calls": count_llm_calls(run),
        "tool_calls": count_tool_calls(run),
        "output": run.outputs
    }

    yield inputs, outputs


@pytest.mark.parametrize("run_suites", _run_suites)
@pytest.mark.parametrize("run_summarizer_function", [
    _custom_run_summarizer,
    "test_langchain_mlrun._custom_run_summarizer",
])
@pytest.mark.parametrize("run_summarizer_settings", [
    _CustomRunSummarizerSettings(dummy_value=12),
    "test_langchain_mlrun._CustomRunSummarizerSettings",
    None,
])
def test_monitor_settings_custom_run_summarizer(
    manual_mode_settings: MLRunTracerSettings,
    run_suites: tuple[Callable, int],
    run_summarizer_function: Callable | str,
    run_summarizer_settings: BaseSettings | str | None,
):
    """
    Test the custom run summarizer that can be passed to MLRunTracer.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param run_suites: The function to run with the expected monitored events.
    :param run_summarizer_function: The custom summarizer function or its import path.
    :param run_summarizer_settings: The settings for the custom summarizer or its import path.
    """
    run_func, _ = run_suites
    manual_mode_settings.monitor.run_summarizer_function = run_summarizer_function
    manual_mode_settings.monitor.run_summarizer_settings = run_summarizer_settings
    dummy_value_for_settings_from_env = 26
    os.environ["TEST_CUSTOM_SUMMARIZER_SETTINGS_DUMMY_VALUE"] = str(dummy_value_for_settings_from_env)

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        print(run_func())
        assert len(tracer.settings.monitor.debug_target_list) == 1

        event = tracer.settings.monitor.debug_target_list[0]
        if run_summarizer_settings:
            if isinstance(run_summarizer_settings, str):
                assert event["input_data"]["input_data"]["from_settings"] == dummy_value_for_settings_from_env
            else:
                assert event["input_data"]["input_data"]["from_settings"] == run_summarizer_settings.dummy_value
        else:
            assert event["input_data"]["input_data"]["from_settings"] == 0


def test_monitor_settings_include_errors_field_presence(manual_mode_settings: MLRunTracerSettings):
    """
    Test that when `include_errors` is True, the error field is present in outputs.
    When `include_errors` is False, the error field is not added to outputs.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    """
    # Run with include_errors=True (default) and verify error field is present:
    manual_mode_settings.monitor.include_errors = True

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        _run_simple_chain()
        assert len(tracer.settings.monitor.debug_target_list) > 0

        for event in tracer.settings.monitor.debug_target_list:
            output_data = event["output_data"]["output_data"]
            assert "error" in output_data, "error field should be present when include_errors is True"

    # Now run with include_errors=False and verify error field is excluded:
    manual_mode_settings.monitor.include_errors = False
    manual_mode_settings.monitor.debug_target_list.clear()

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        _run_simple_chain()
        assert len(tracer.settings.monitor.debug_target_list) > 0

        for event in tracer.settings.monitor.debug_target_list:
            output_data = event["output_data"]["output_data"]
            assert "error" not in output_data, "error field should be excluded when include_errors is False"


def test_monitor_settings_include_full_run(manual_mode_settings: MLRunTracerSettings):
    """
    Test that when `include_full_run` is True, the complete serialized run is included in outputs.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    """
    manual_mode_settings.monitor.include_full_run = True

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        _run_simple_chain()

        assert len(tracer.settings.monitor.debug_target_list) > 0

        for event in tracer.settings.monitor.debug_target_list:
            output_data = event["output_data"]["output_data"]
            assert "full_run" in output_data, "full_run should be included in outputs when include_full_run is True"
            # Verify the full_run contains expected run structure:
            assert "inputs" in output_data["full_run"]
            assert "outputs" in output_data["full_run"]


def test_monitor_settings_include_metadata(manual_mode_settings: MLRunTracerSettings):
    """
    Test that when `include_metadata` is False, metadata is excluded from inputs.

    Note: The fake models used in tests don't produce runs with metadata, so we can only
    verify the "exclude" behavior. The code only adds metadata if the run actually contains it.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    """
    # Run with include_metadata=False and verify metadata is excluded:
    manual_mode_settings.monitor.include_metadata = False

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        _run_simple_chain()
        assert len(tracer.settings.monitor.debug_target_list) > 0

        # Check that metadata is not present in inputs:
        for event in tracer.settings.monitor.debug_target_list:
            input_data = event["input_data"]["input_data"]
            assert "metadata" not in input_data, "metadata should be excluded when include_metadata is False"


def test_monitor_settings_include_latency(manual_mode_settings: MLRunTracerSettings):
    """
    Test that when `include_latency` is False, latency is excluded from outputs.

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    """
    manual_mode_settings.monitor.include_latency = False

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        _run_simple_chain()
        assert len(tracer.settings.monitor.debug_target_list) > 0

        for event in tracer.settings.monitor.debug_target_list:
            assert "latency" not in event["output_data"]["output_data"], \
                "latency should be excluded when include_latency is False"


def test_import_from_module_path_errors():
    """
    Test that `_import_from_module_path` raises appropriate errors for invalid paths.
    """
    # Test ValueError for path without a dot:
    with pytest.raises(ValueError) as exc_info:
        MLRunTracer._import_from_module_path("no_dot_path")
    assert "must have at least one '.'" in str(exc_info.value)

    # Test ImportError for non-existent module:
    with pytest.raises(ImportError) as exc_info:
        MLRunTracer._import_from_module_path("nonexistent_module_xyz.SomeClass")
    assert "Could not import" in str(exc_info.value)

    # Test AttributeError for non-existent attribute in existing module:
    with pytest.raises(AttributeError) as exc_info:
        MLRunTracer._import_from_module_path("os.nonexistent_attribute_xyz")
    assert "Could not import" in str(exc_info.value)


#: Sample structured runs for testing LangChainMonitoringApp methods.
_sample_structured_runs = [
    {
        "label": "test_label",
        "child_level": 0,
        "input_data": {
            "run_name": "RunnableSequence",
            "run_type": "chain",
            "tags": ["tag1"],
            "inputs": {"topic": "MLRun"},
            "start_timestamp": "2024-01-01T10:00:00+00:00",
        },
        "output_data": {
            "outputs": {"result": "test output"},
            "end_timestamp": "2024-01-01T10:00:01+00:00",
            "error": None,
            "child_runs": [
                {
                    "input_data": {
                        "run_name": "FakeListChatModel",
                        "run_type": "llm",
                        "tags": ["tag2"],
                        "inputs": {"prompt": "test"},
                        "start_timestamp": "2024-01-01T10:00:00.100+00:00",
                    },
                    "output_data": {
                        "outputs": {
                            "generations": [[{
                                "message": {
                                    "kwargs": {
                                        "usage_metadata": {
                                            "input_tokens": 10,
                                            "output_tokens": 20,
                                        }
                                    }
                                }
                            }]]
                        },
                        "end_timestamp": "2024-01-01T10:00:00.500+00:00",
                        "error": None,
                    },
                },
            ],
        },
    },
    {
        "label": "test_label",
        "child_level": 0,
        "input_data": {
            "run_name": "SimpleAgent",
            "run_type": "chain",
            "tags": ["tag1"],
            "inputs": {"query": "test query"},
            "start_timestamp": "2024-01-01T10:00:02+00:00",
        },
        "output_data": {
            "outputs": {"result": "agent output"},
            "end_timestamp": "2024-01-01T10:00:04+00:00",
            "error": "SomeError: something went wrong",
        },
    },
]


def test_langchain_monitoring_app_iterate_structured_runs():
    """
    Test that `iterate_structured_runs` yields all runs including nested child runs.
    """
    # Iterate over all runs:
    all_runs = list(LangChainMonitoringApp.iterate_structured_runs(_sample_structured_runs))

    # Should yield parent runs and child runs:
    # - First sample: 1 parent + 1 child = 2 runs
    # - Second sample: 1 parent = 1 run
    # Total: 3 runs
    assert len(all_runs) == 3

    # Verify run names are as expected:
    run_names = [r["input_data"]["run_name"] for r in all_runs]
    assert "RunnableSequence" in run_names
    assert "FakeListChatModel" in run_names
    assert "SimpleAgent" in run_names


def test_langchain_monitoring_app_count_run_names():
    """
    Test that `count_run_names` correctly counts occurrences of each run name.
    """
    counts = LangChainMonitoringApp.count_run_names(_sample_structured_runs)

    assert counts["RunnableSequence"] == 1
    assert counts["FakeListChatModel"] == 1
    assert counts["SimpleAgent"] == 1


def test_langchain_monitoring_app_count_token_usage():
    """
    Test that `count_token_usage` correctly calculates total tokens from LLM runs.
    """
    token_usage = LangChainMonitoringApp.count_token_usage(_sample_structured_runs)

    assert token_usage["total_input_tokens"] == 10
    assert token_usage["total_output_tokens"] == 20
    assert token_usage["combined_total"] == 30


def test_langchain_monitoring_app_calculate_success_rate():
    """
    Test that `calculate_success_rate` returns the correct percentage of successful runs.
    """
    success_rate = LangChainMonitoringApp.calculate_success_rate(_sample_structured_runs)

    # First run has no error, second run has error:
    # Success rate should be 1/2 = 0.5
    assert success_rate == 0.5

    # Test with empty list:
    empty_rate = LangChainMonitoringApp.calculate_success_rate([])
    assert empty_rate == 0.0

    # Test with all successful runs:
    successful_runs = [_sample_structured_runs[0]]  # Only the first run which has no error
    all_success_rate = LangChainMonitoringApp.calculate_success_rate(successful_runs)
    assert all_success_rate == 1.0


def test_langchain_monitoring_app_calculate_average_latency():
    """
    Test that `calculate_average_latency` returns the correct average latency across root runs.
    """
    # Calculate average latency:
    avg_latency = LangChainMonitoringApp.calculate_average_latency(_sample_structured_runs)

    # First run: 10:00:00 to 10:00:01 = 1000ms
    # Second run: 10:00:02 to 10:00:04 = 2000ms
    # Average: (1000 + 2000) / 2 = 1500ms
    assert avg_latency == 1500.0

    # Test with empty list:
    empty_latency = LangChainMonitoringApp.calculate_average_latency([])
    assert empty_latency == 0.0


def test_langchain_monitoring_app_calculate_average_latency_skips_child_runs():
    """
    Test that `calculate_average_latency` skips child runs (only calculates for root runs).
    """
    # Create a sample with a child run that has child_level > 0:
    runs_with_child = [
        {
            "label": "test",
            "child_level": 0,
            "input_data": {"start_timestamp": "2024-01-01T10:00:00+00:00"},
            "output_data": {"end_timestamp": "2024-01-01T10:00:01+00:00"},
        },
        {
            "label": "test",
            "child_level": 1,  # This is a child run, should be skipped
            "input_data": {"start_timestamp": "2024-01-01T10:00:00+00:00"},
            "output_data": {"end_timestamp": "2024-01-01T10:00:10+00:00"},  # 10 seconds - would skew average
        },
    ]

    # Calculate average latency:
    avg_latency = LangChainMonitoringApp.calculate_average_latency(runs_with_child)

    # Should only consider the root run (1000ms), not the child run:
    assert avg_latency == 1000.0


def test_debug_mode_stdout(manual_mode_settings: MLRunTracerSettings, capsys):
    """
    Test that debug mode prints to stdout when `debug_target_list` is not set (is False).

    :param manual_mode_settings: Fixture to set up 'manual' mode environment and settings.
    :param capsys: Pytest fixture to capture stdout/stderr.
    """
    # Set debug mode with debug_target_list=False (should print to stdout):
    manual_mode_settings.monitor.debug = True
    manual_mode_settings.monitor.debug_target_list = False

    with mlrun_monitoring(settings=manual_mode_settings) as tracer:
        _run_simple_chain()

    # Capture stdout:
    captured = capsys.readouterr()

    # Verify that JSON output was printed to stdout:
    assert "event_id" in captured.out, "Event should be printed to stdout when debug_target_list is False"
    assert "input_data" in captured.out
    assert "output_data" in captured.out
