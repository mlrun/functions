# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess
from typing import List, Union, Optional
import sys
import click
import yaml
import re

from cli.helpers import (
    is_item_dir,
    install_pipenv,
    install_python,
    install_requirements,
    get_item_yaml_values,
)
from cli.path_iterator import PathIterator


@click.command()
@click.option("-r", "--root-directory", default=".", help="Path to root directory")
@click.option("-s", "--suite", help="Type of suite to run [py/ipynb/examples/items]")
@click.option("-mp", "--multi-processing", help="run multiple tests")
@click.option("-fn", "--function-name", help="run specific function by name")
@click.option(
    "-f",
    "--stop-on-failure",
    is_flag=True,
    default=False,
    help="When true, test suite will stop running after the first test ran",
)
def test_suite(root_directory: str,
               suite: str,
               stop_on_failure: bool,
               multi_processing: bool = False,
               function_name: str = None):
    if not suite:
        click.echo("-s/--suite is required")
        exit(1)

    if suite == "py":
        TestPY(stop_on_failure=stop_on_failure, clean_env_artifacts=True)._run(
            root_directory, multi_processing, function_name
        )
    elif suite == "ipynb":
        TestIPYNB(stop_on_failure=stop_on_failure, clean_env_artifacts=True)._run(
            root_directory, multi_processing, function_name
        )
    elif suite == "examples":
        test_example(root_directory)
    elif suite == "items":
        TestItemYamls(stop_on_failure=stop_on_failure)
    else:
        click.echo(f"Suite {suite} is unsupported")
        exit(1)


def test_example(root_dir="."):
    # install_pipenv()
    #
    # for directory in PathIterator(root=root_dir, rule=is_item_dir):
    #     notebooks = [n for n in PathIterator(root=directory, rule=is_test_notebook)]
    #     if not notebooks:
    #         continue
    #
    #     # install_python(directory)
    #     item_requirements = list(get_item_yaml_values(directory, 'requirements')['requirements'])
    #     install_requirements(directory, ["papermill", "jupyter"] + item_requirements)
    #
    #     # for notebook in notebooks:
    #     #     print(f"Running tests for {notebook}...")
    #     #     run_papermill: CompletedProcess = subprocess.run(
    #     #         ["pipenv", "run", "papermill", notebook, "-"],
    #     #         stdout=subprocess.PIPE,
    #     #         stderr=subprocess.PIPE,
    #     #         cwd=directory,
    #     #     )
    #     #     if run_papermill.returncode != 0:
    #     #         print_std(run_papermill)
    #     #         exit(run_papermill.returncode)
    #     #     exit(0)
    pass


@dataclass
class TestResult:
    status: str
    status_code: Optional[int]
    meta_data: dict = field(default_factory=dict)

    @classmethod
    def passed(
            cls, status_code: Optional[int] = None, meta_data: Optional[dict] = None
    ):
        return cls(status="Passed", status_code=status_code, meta_data=meta_data)

    @classmethod
    def failed(
            cls, status_code: Optional[int] = None, meta_data: Optional[dict] = None
    ):
        return cls(status="Failed", status_code=status_code, meta_data=meta_data)

    @classmethod
    def ignored(
            cls, status_code: Optional[int] = None, meta_data: Optional[dict] = None
    ):
        return cls(status="Ignored", status_code=status_code, meta_data=meta_data)


class TestSuite(ABC):
    def __init__(self, stop_on_failure: bool = True):
        self.stop_on_failure = stop_on_failure
        self.test_results = []

    @abstractmethod
    def discover(self, path: Union[str, Path]) -> List[str]:
        pass

    @abstractmethod
    def run(self, path: Union[str, Path]) -> TestResult:
        pass

    @abstractmethod
    def before_run(self):
        pass

    @abstractmethod
    def after_run(self):
        pass

    @abstractmethod
    def before_each(self, path: Union[str, Path]):
        pass

    @abstractmethod
    def after_each(self, path: Union[str, Path], test_result: TestResult):
        pass

    def _run(self, path: Union[str, Path], multiprocess, function_name):
        import multiprocessing as mp
        process_count = 1
        if multiprocess:
            process_count = mp.cpu_count() - 1
        print("running tests with {} process".format(process_count))
        discovered_functions = self.discover(path)
        if function_name is not None:
            discovered_functions = [fn for fn in discovered_functions if function_name == Path(fn).stem]
        for path in discovered_functions:
            if re.match(".+/test_*", path):
                discovered_functions.remove(path)
                print("a function name cannot start with test, please rename {} ".format(path))

        self.before_run()

        # pool = mp.Pool(process_count)
        # pool.map(self.directory_process, [directory for directory in discovered])
        for directory in discovered_functions:
            self.directory_process(directory)
        self.after_run()

        # pool.close()
        sys.exit(0)

    def directory_process(self, directory):
        self.before_each(directory)
        result = self.run(directory)
        self.test_results.append(result)
        self.after_each(directory, result)


class TestPY(TestSuite):
    def __init__(self, stop_on_failure: bool = True, clean_env_artifacts: bool = True):
        super().__init__(stop_on_failure)
        self.clean_env_artifacts = clean_env_artifacts
        self.results = []

    def discover(self, path: Union[str, Path]) -> List[str]:
        path = Path(path)
        testable = []
        item_yaml_path = path / "item.yaml"
        # Handle single test file
        if item_yaml_path.exists():
            for inner_file in path.iterdir():
                if self.is_test_py(inner_file):
                    if is_test_valid_by_item(path):
                        testable.append(str(path.resolve()))
                        break
            if testable:
                click.echo("Found testable directory...")
        # Handle multiple directories
        else:
            item_iterator = PathIterator(root=path, rule=is_item_dir, as_path=True)
            for inner_dir in item_iterator:
                # Iterate individual files in each directory
                for inner_file in inner_dir.iterdir():
                    if self.is_test_py(inner_file):
                        if is_test_valid_by_item(inner_dir):
                            testable.append(str(inner_dir.resolve()))
                            break
            click.echo(f"Found {len(testable)} testable items...")

        if not testable:
            click.echo(
                "No tests found, make sure your test file names are structures as 'test_*.py')"
            )
            sys.exit(0)
        testable.sort()
        print(testable)
        return testable

    def before_run(self):
        install_pipenv()

    def before_each(self, path: Union[str, Path]):
        pass

    def run(self, path: Union[str, Path]):
        print("PY run path {}".format(path))
        install_python(path)
        item_requirements = list(get_item_yaml_values(path, 'requirements')['requirements'])
        mlrun_version = list(get_item_yaml_values(path, "mlrunVersion")["mlrunVersion"])[0]
        install_requirements(path, ["pytest", f"mlrun=={mlrun_version}"] + item_requirements)
        click.echo(f"Running tests for {path}...")
        completed_process: CompletedProcess = subprocess.run(
            f"cd {path} ; pipenv run python -m pytest",
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            cwd=path,
            shell=True,
        )

        meta_data = {"completed_process": completed_process, "test_path": path}

        if completed_process.returncode == 0:
            return TestResult.passed(status_code=0, meta_data=meta_data)

        return TestResult.failed(
            status_code=completed_process.returncode,
            meta_data=meta_data,
        )

    def after_each(self, path: Union[str, Path], test_result: TestResult):
        if self.clean_env_artifacts:
            clean_pipenv(path)

        if self.stop_on_failure:
            if test_result.status == "Failed":
                complete_subprocess: CompletedProcess = test_result.meta_data[
                    "complete_subprocess"
                ]
                click.echo(f"{path} [Failed]")
                click.echo("==================== stdout ====================")
                click.echo(complete_subprocess.stdout.decode("utf-8"))
                click.echo("==================== stderr ====================")
                click.echo(complete_subprocess.stderr.decode("utf-8"))
                exit(test_result.status_code)

    def after_run(self):
        failed_tests = []
        passed_tests = []
        ignored_tests = []

        for test_result in self.test_results:
            if test_result.status == "Failed":
                failed_tests.append(test_result)
            elif test_result.status == "Passed":
                passed_tests.append(test_result)
            elif test_result.status == "Ignored":
                ignored_tests.append(ignored_tests)
            else:
                click.echo(f"Unsupported test status detected: {test_result.status}")

        click.echo(f"Passed: {len(passed_tests)}")
        click.echo(f"Failed: {len(failed_tests)}")
        click.echo(f"Ignored: {len(ignored_tests)}")

        for passed_test in passed_tests:
            test_path = passed_test.meta_data["test_path"]
            click.echo(f"{test_path} [Passed]")

        for ignore_test in ignored_tests:
            test_path = ignore_test.meta_data["test_path"]
            click.echo(f"{test_path} [Ignored]")

        for failed_test in failed_tests:
            test_path = failed_test.meta_data["test_path"]
            process = failed_test.meta_data["completed_process"]
            click.echo(f"{test_path} [Failed]")
            click.echo("==================== stdout ====================")
            if process.stdout:
                click.echo(process.stdout.decode("utf-8"))
            click.echo("==================== stderr ====================")
            click.echo(process.stderr.decode("utf-8"))
            click.echo("\n")

        if failed_tests:
            sys.exit(1)

    @staticmethod
    def is_test_py(path: Union[str, Path]) -> bool:
        return (
                path.is_file()
                and path.name.startswith("test_")
                and path.name.endswith(".py")
        )


class TestIPYNB(TestSuite):
    def __init__(self, stop_on_failure: bool = True, clean_env_artifacts: bool = True):
        super().__init__(stop_on_failure)
        self.clean_env_artifacts = clean_env_artifacts
        self.results = []

    def discover(self, path: Union[str, Path]) -> List[str]:
        path = Path(path)
        testables = []

        # Handle single test directory
        if (path / "item.yaml").exists():
            for inner_file in path.iterdir():
                if self.is_test_ipynb(inner_file):
                    testables.append(str(inner_file.resolve()))
            if testables:
                click.echo("Found testable directory...")
        # Handle multiple directories
        else:
            item_iterator = PathIterator(root=path, rule=is_item_dir, as_path=True)
            for inner_dir in item_iterator:
                # Iterate individual files in each directory
                for inner_file in inner_dir.iterdir():
                    # click.echo("test inner file"+str(inner_file))
                    if self.is_test_ipynb(inner_file):
                        # click.echo("adding "+str(inner_file))
                        testables.append(str(inner_dir.resolve()))
            click.echo(f"Found {len(testables)} testable items...")

        if not testables:
            click.echo(
                "No tests found, make sure your test file names are structures as 'test_*.py')"
            )
            exit(0)
        testables.sort()
        click.echo(
            "tests list " + str(testables)
        )
        return testables

    def before_run(self):
        install_pipenv()

    def before_each(self, path: Union[str, Path]):
        pass

    #    def run(self, path: Union[str, Path]) -> TestResult:
    def run(self, path: Union[str, Path]) -> TestResult:
        print("IPYNB run path {}".format(path))
        install_python(path)
        item_requirements = list(get_item_yaml_values(path, 'requirements')['requirements'])
        install_requirements(path, ["papermill"] + item_requirements)

        click.echo(f"Running tests for {path}...")
        running_ipynb = Path(path).name + ".ipynb"
        click.echo(f"Running notebook {running_ipynb}")
        command = f'pipenv run papermill {running_ipynb} out.ipynb --log-output'
        completed_process: CompletedProcess = subprocess.run(
            f"cd {path} ;echo {command} ; {command}",
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            cwd=path,
            shell=True
        )

        meta_data = {"completed_process": completed_process, "test_path": path}
        click.echo(completed_process)
        if completed_process.returncode == 0:
            return TestResult.passed(status_code=0, meta_data=meta_data)

        return TestResult.failed(
            status_code=completed_process.returncode,
            meta_data=meta_data,
        )

    def after_run(self):
        failed_tests = []
        passed_tests = []
        ignored_tests = []

        for test_result in self.test_results:
            if test_result.status == "Failed":
                failed_tests.append(test_result)
            elif test_result.status == "Passed":
                passed_tests.append(test_result)
            elif test_result.status == "Ignored":
                ignored_tests.append(ignored_tests)
            else:
                click.echo(f"Unsupported test status detected: {test_result.status}")

        click.echo(f"Passed: {len(passed_tests)}")
        click.echo(f"Failed: {len(failed_tests)}")
        click.echo(f"Ignored: {len(ignored_tests)}")

        for passed_test in passed_tests:
            test_path = passed_test.meta_data["test_path"]
            click.echo(f"{test_path} [Passed]")

        for ignore_test in ignored_tests:
            test_path = ignore_test.meta_data["test_path"]
            click.echo(f"{test_path} [Ignored]")

        for failed_test in failed_tests:
            test_path = failed_test.meta_data["test_path"]
            process = failed_test.meta_data["completed_process"]
            click.echo(f"{test_path} [Failed]")
            click.echo("==================== stdout ====================")
            if process.stdout is not None:
                click.echo(process.stdout.decode("utf-8"))
            click.echo("==================== stderr ====================")
            click.echo(process.stderr.decode("utf-8"))
            click.echo("\n")

        if failed_tests:
            exit(1)

    def after_each(self, path: Union[str, Path], test_result: TestResult):
        if self.clean_env_artifacts:
            clean_pipenv(path)

        if self.stop_on_failure:
            if test_result.status == "Failed":
                complete_subprocess: CompletedProcess = test_result.meta_data[
                    "complete_subprocess"
                ]
                click.echo(f"{path} [Failed]")
                click.echo("==================== stdout ====================")
                click.echo(complete_subprocess.stdout.decode("utf-8"))
                click.echo("==================== stderr ====================")
                click.echo(complete_subprocess.stderr.decode("utf-8"))
                exit(test_result.status_code)

    def _run(self, path: Union[str, Path], multi_processing, function_name):
        super()._run(path, multi_processing, function_name)

    @staticmethod
    def is_test_ipynb(path: Path):
        return (
                path.is_file()
                and path.name.endswith(".ipynb")
        )


class TestItemYamls(TestSuite):
    def __init__(self, stop_on_failure: bool = True):
        super().__init__(stop_on_failure)

    def discover(self, path: Union[str, Path]) -> List[str]:
        path = Path(path)
        testables = []

        # Handle single test directory
        if (path / "item.yaml").exists():
            testables.append(str((path / "item.yaml").resolve()))
            if testables:
                click.echo("Found testable directory...")
        # Handle multiple directories
        else:
            item_iterator = PathIterator(root=path, rule=is_item_dir, as_path=True)
            for item_dir in item_iterator:
                testables.append(str((item_dir / "item.yaml").resolve()))
            click.echo(f"Found {len(testables)} testable items...")

        if not testables:
            click.echo(
                "No tests found, make sure your test file names are structures as 'test_*.py')"
            )
            exit(0)

        return testables

    def run(self, path: Union[str, Path]) -> TestResult:
        path = Path(path)
        item = yaml.full_load(open(path, "r"))
        directory = path.parent

        if item.get("spec")["filename"]:
            implementation_file = directory / item.get("spec")["filename"]
            if not implementation_file.exists():
                return TestResult.failed(
                    status_code=1,
                    meta_data={
                        "message": f"Item.spec.filename ({implementation_file}) not found",
                        "test_path": path,
                    },
                )
        if item["example"]:
            example_file = directory / item["example"]
            if not example_file.exists():
                return TestResult.failed(
                    status_code=1,
                    meta_data={
                        "message": f"Item.example ({example_file}) not found",
                        "test_path": path,
                    },
                )

        if item["doc"]:
            doc_file = directory / item["doc"]
            if not doc_file.exists():
                return TestResult.failed(
                    status_code=1,
                    meta_data={
                        "message": f"Item.doc ({doc_file}) not found",
                        "test_path": path,
                    },
                )

        return TestResult.passed(
            status_code=0,
            meta_data={"message": f"Valid item {path}", "test_path": path},
        )

    def before_run(self):
        pass

    def after_run(self):
        failed_tests = []
        passed_tests = []
        ignored_tests = []

        for test_result in self.test_results:
            if test_result.status == "Failed":
                failed_tests.append(test_result)
            elif test_result.status == "Passed":
                passed_tests.append(test_result)
            elif test_result.status == "Ignored":
                ignored_tests.append(ignored_tests)
            else:
                click.echo(f"Unsupported test status detected: {test_result.status}")

        click.echo(f"Passed: {len(passed_tests)}")
        click.echo(f"Failed: {len(failed_tests)}")
        click.echo(f"Ignored: {len(ignored_tests)}")

        for passed_test in passed_tests:
            test_path = passed_test.meta_data["test_path"]
            click.echo(f"{test_path} [Passed]")

        for ignore_test in ignored_tests:
            test_path = ignore_test.meta_data["test_path"]
            click.echo(f"{test_path} [Ignored]")

        for failed_test in failed_tests:
            test_path = failed_test.meta_data["test_path"]
            click.echo(f"{test_path} [Failed]")

        if failed_tests:
            exit(1)

    def before_each(self, path: Union[str, Path]):
        pass

    def after_each(self, path: Union[str, Path], test_result: TestResult):
        if self.stop_on_failure:
            if test_result.status == "Failed":
                message = test_result.meta_data["message"]
                click.echo(f"{path} [Failed]")
                click.echo(f"Error: {message}")
                exit(1)

    def _run(self, path: Union[str, Path]):
        super()._run(path)


def clean_pipenv(directory: str):
    pip_file = Path(directory) / "Pipfile"
    pip_lock = Path(directory) / "Pipfile.lock"

    if pip_file.exists():
        pip_file.unlink()
    if pip_lock.exists():
        pip_lock.unlink()


# load item yaml
def load_item(path):
    with open(path, 'r') as stream:
        data = yaml.load(stream)
    return data


def is_test_valid_by_item(item_posix_path):
    full_path = str(item_posix_path.absolute())+'/item.yaml'
    data = load_item(full_path)
    if data.get("test_valid") is not None:
        test_valid = data.get("test_valid")
        test_name = data.get("name")
        if not test_valid:
            click.echo("==================== Test {} Not valid ====================".format(test_name))
            click.echo("==================== enable tet_valid in item yaml ====================")
        return test_valid
    else:
        return True


if __name__ == "__main__":
    test_suite()
