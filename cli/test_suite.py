import subprocess
from pathlib import Path

import click
import yaml

from cli.helpers import (
    is_item_dir,
    install_pipenv,
    install_python,
    install_requirements,
    get_item_yaml_requirements,
)
from cli.path_iterator import PathIterator


@click.command()
@click.option("-r", "--root-directory", default=".", help="Path to root directory")
@click.option("-s", "--suite", help="Type of suite to run [py/ipynb/examples/items]")
def test_suite(root_directory: str, suite: str):
    if not suite:
        click.echo("-s/--suite is required")
        exit(1)

    if suite == "py":
        test_py(root_directory, clean=True)
    elif suite == "ipynb":
        test_ipynb(root_directory, clean=True)
    elif suite == "examples":
        test_example(root_directory)
    elif suite == "items":
        test_item_files(root_directory)
    else:
        click.echo(f"Suite {suite} is unsupported")
        exit(1)


def test_py(root_dir=".", clean=False):
    click.echo("Collecting items...")

    item_iterator = PathIterator(root=root_dir, rule=is_item_dir, as_path=True)
    testable_items = []

    for item_dir in item_iterator:
        testable = any(
            (
                i.name.startswith("test_") and i.name.endswith(".py")
                for i in item_dir.iterdir()
            )
        )
        if testable:
            testable_items.append(item_dir)

    click.echo(f"Found {len(testable_items)} testable items...")

    if not testable_items:
        exit(0)

    install_pipenv()

    for directory in testable_items:
        directory = directory.resolve()

        install_python(directory)
        item_requirements = get_item_yaml_requirements(directory)
        install_requirements(directory, ["pytest"] + item_requirements)

        print(f"Running tests for {directory}...")

        run_tests: subprocess.CompletedProcess = subprocess.run(
            f"cd {directory} ; pipenv run python -m pytest",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=directory,
            shell=True,
        )

        print_std(run_tests)

        if clean:
            clean_pipenv(directory)

        if run_tests.returncode != 0:
            exit(run_tests.returncode)


def test_ipynb(root_dir=".", clean=False):
    click.echo("Collecting items...")

    item_iterator = PathIterator(root=root_dir, rule=is_item_dir, as_path=True)
    testable_items = []

    for item_dir in item_iterator:
        testable = any(
            (
                i.name.startswith("test_") and i.name.endswith(".ipynb")
                for i in item_dir.iterdir()
            )
        )
        if testable:
            testable_items.append(item_dir)

    click.echo(f"Found {len(testable_items)} testable items...")

    if not testable_items:
        exit(0)

    install_pipenv()

    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        notebooks = [n for n in PathIterator(root=directory, rule=is_test_notebook)]
        if not notebooks:
            continue

        install_python(directory)
        item_requirements = get_item_yaml_requirements(directory)
        install_requirements(directory, ["papermill", "jupyter"] + item_requirements)

        for notebook in notebooks:
            print(f"Running tests for {notebook}...")
            run_papermill: subprocess.CompletedProcess = subprocess.run(
                ["pipenv", "run", "papermill", notebook, "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=directory,
            )
            if run_papermill.returncode != 0:
                print_std(run_papermill)
                exit(run_papermill.returncode)

        if clean:
            clean_pipenv(directory)

        exit(0)


def test_example(root_dir="."):
    install_pipenv()

    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        notebooks = [n for n in PathIterator(root=directory, rule=is_test_notebook)]
        if not notebooks:
            continue

        # install_python(directory)
        item_requirements = get_item_yaml_requirements(directory)
        install_requirements(directory, ["papermill", "jupyter"] + item_requirements)

        # for notebook in notebooks:
        #     print(f"Running tests for {notebook}...")
        #     run_papermill: subprocess.CompletedProcess = subprocess.run(
        #         ["pipenv", "run", "papermill", notebook, "-"],
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.PIPE,
        #         cwd=directory,
        #     )
        #     if run_papermill.returncode != 0:
        #         print_std(run_papermill)
        #         exit(run_papermill.returncode)
        #     exit(0)


def test_item_files(root_dir="."):
    click.echo("Collecting items...")

    item_iterator = PathIterator(root=root_dir, rule=is_item_dir, as_path=True)
    items = [i / "item.yaml" for i in item_iterator]

    click.echo(f"Found {len(items)} items...")

    error = False
    for item_path in items:
        try:
            directory = item_path.parent

            with open(item_path, "r") as f:
                item = yaml.full_load(f)

            if item.get("spec")["filename"]:
                implementation_file = directory / item.get("spec")["filename"]
                if not implementation_file.exists():
                    raise FileNotFoundError(implementation_file)

            if item["example"]:
                example_file = directory / item["example"]
                if not example_file.exists():
                    raise FileNotFoundError(example_file)

            if item["doc"]:
                doc_file = directory / item["doc"]
                if not doc_file.exists():
                    raise FileNotFoundError(doc_file)

            click.secho(f"{item_path} ", nl=False)
            click.secho(f"[Passed]", fg="green")
        except FileNotFoundError as e:
            error = True
            click.secho(f"{item_path} ", nl=False)
            click.secho(f"[Failed]:", fg="red")
            click.echo(str(e))

    if error:
        exit(1)


def clean(root_dir="."):
    for directory in PathIterator(root=root_dir, rule=is_item_dir):
        clean_pipenv(directory)


def is_test_notebook(path: Path) -> bool:
    return (
        path.is_file() and path.name.startswith("test") and path.name.endswith(".ipynb")
    )


def is_example_notebook(path: Path) -> bool:
    return (
        path.is_file()
        and not path.name.startswith("test")
        and path.name.endswith(".ipynb")
    )


def print_std(subprocess_result):
    print()
    print("==================== stdout ====================")
    print(subprocess_result.stdout.decode("utf-8"))
    print("==================== stderr ====================")
    print(subprocess_result.stderr.decode("utf-8"))
    print()


def clean_pipenv(directory: str):
    pip_file = Path(directory) / "Pipfile"
    pip_lock = Path(directory) / "Pipfile.lock"

    if pip_file.exists():
        pip_file.unlink()
    if pip_lock.exists():
        pip_lock.unlink()


if __name__ == "__main__":
    test_suite()
