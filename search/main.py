import os
import subprocess
import sys

import click


@click.command()
@click.option(
    "--formula", type=str, prompt="Enter ion formula", help="Ion formula name"
)
@click.option("--charge", type=str, prompt="Enter ion charge", help="Ion charge")
@click.option(
    "--path",
    type=str,
    default="",
    prompt="Enter path to the index dictionaries directory",
    help="Path to index dictionaries directory",
)
@click.option(
    "--n_jobs",
    type=str,
    default="1",
    prompt="Enter number of jobs",
    help="Maximum number of concurrently running workers",
)
@click.option(
    "--report_name",
    type=str,
    default="",
    prompt="Enter report name",
    help="Directory, where report will be saved",
)
@click.option(
    "--autoclb",
    type=str,
    default="No",
    prompt="Use Automatic Calibration?",
    help='If "Yes", auto-calibration algorithm will be preformed',
)
@click.option(
    "--threshold",
    type=str,
    default="auto",
    prompt="Enter cosine distance threshold",
    help="Measure of search selectivity.",
)
@click.option(
    "--fp_filter",
    type=str,
    default="Yes",
    prompt="Use False Positive Filter?",
    help='If "Yes", false positive filtering model will be performed.',
)
@click.option(
    "--window_size",
    type=str,
    default="0",
    prompt="Set window size",
    help="If not 0, sliding windows will be performed.",
)
@click.option(
    "--make_plots",
    type=str,
    default="Yes",
    prompt="Make plots?",
    help='If "Yes", plotting will be performed.',
)
def search(
    formula,
    charge,
    path,
    n_jobs,
    report_name,
    autoclb,
    threshold,
    fp_filter,
    window_size,
    make_plots,
):
    subprocess.call(
        [
            "./search.sh",
            formula,
            charge,
            path,
            n_jobs,
            report_name,
            autoclb,
            threshold,
            fp_filter,
            window_size,
            make_plots,
        ]
    )


@click.command()
@click.option(
    "--shard_number",
    type=str,
    prompt="Enter number of shards",
    help="Number of shards used for search parallelization",
    default="10",
)
@click.option(
    "--upper_limit_size",
    type=str,
    prompt="Set limit file size for indexing spectra",
    help="File weight limit to avoid server overload",
    default="1000M",
)
@click.option(
    "--inner_limit_size",
    type=str,
    prompt="Enter inner limit size",
    help="If file is less than inner_limit_size, it will not be indexing",
    default="5M",
)
def create_batches(shard_number, upper_limit_size, inner_limit_size):
    subprocess.call(
        ["./create_batches.sh", shard_number, upper_limit_size, inner_limit_size]
    )


@click.command()
@click.option(
    "--word_indicator",
    type=str,
    prompt="Set word indicator",
    help="The word, which should be contained in spectrum filename to decrease number of spectra.",
)
@click.option(
    "--batch_name",
    type=str,
    prompt="Enter batch name",
    help="Text filename, which will contain" " batched paths to spectra",
)
@click.option(
    "--upper_limit_size",
    type=str,
    prompt="Enter upper limit size",
    help="If file is bigger than upper_limit_size, it will not be indexing",
    default="1000M",
)
@click.option(
    "--inner_limit_size",
    type=str,
    prompt="Enter inner limit size",
    help="If file is less than inner_limit_size, it will not be indexing",
    default="5M",
)
def create_unique_batch(word_indicator, batch_name, upper_limit_size, inner_limit_size):
    subprocess.call(
        [
            "./create_unique_batch.sh",
            word_indicator,
            batch_name,
            upper_limit_size,
            inner_limit_size,
        ]
    )


@click.command()
@click.option(
    "--window_size",
    type=str,
    prompt="Set sliding window size (Set 0 if you don't want any windows)",
    help="The parameter, which is number of summarized spectra in one experiment (Need to process monitoring"
    " spectra)",
    default="60",
)
@click.option(
    "--min_distance",
    type=str,
    prompt="Set min. distance between indexed peaks",
    help="The pearameter, which defines minimal m/z difference between peaks in indexed dictionaries",
    default="0.001",
)
def index(window_size, min_distance):
    subprocess.call(["./index.sh", window_size, min_distance])


@click.command()
@click.option(
    "--report_name",
    type=str,
    default="",
    prompt="Enter report name",
    help="Directory, where report is saved",
)
def check_report(report_name):
    subprocess.call(["./check_report.sh", report_name])


@click.command()
@click.option(
    "--task",
    prompt="What do you want to do?",
    help="Choose the action (search, index, create_batches)",
)
def start_app(task):
    pic = (
        "..........................................................................................\n"
        + "....@@@@@@...@@@@@@..@@@@@@@@@@@..@@@@@@@@@....@@@@....@@@@....@@@@@@@.......@@@@@........\n"
        + "....@@@@@@...@@@@@@..@@@@@@@@@@@..@@@@@@@@@@...@@@@@...@@@@..@@@@@@@@@@@....@@@@@@@.......\n"
        + "....@@@@@@@.@@@@@@@..@@@....@@@@..@@@....@@@@..@@@@@...@@@@..@@@@@@........@@@@.@@@@......\n"
        + "....@@@@@@@@@@@@@@@..@@@@@@@@@@@..@@@....@@@@..@@@@@...@@@@...@@@@@@@@@....@@@...@@@......\n"
        + "....@@@@.@@@@@.@@@@..@@@@.........@@@....@@@@..@@@@@..@@@@@........@@@@@..@@@@@@@@@@......\n"
        + "....@@@@.:@@@@.@@@@..@@@@@@@@@....@@@@@@@@@@...@@@@@@@@@@@..@@@@@..@@@@@..@@@@@@@@@@@.....\n"
        + "....@@@@..@@@..@@@@..@@@@@@@@@@@..@@@@@@@@@......@@@@@@@@....@@@@@@@@@@...@@@@....@@@@....\n"
        + "..........................................................................................\n"
    )
    click.echo(pic)
    if task.lower() in ["search", "find", "query", "search_ion", "s"]:
        click.echo(f"Prepare to searching.")
        search()

    elif task.lower() in [
        "create_batches",
        "batch_create",
        "create-batches",
        "make_batches",
        "batches",
        "cb",
    ]:
        click.echo(f"Prepare to all batches creation.")
        create_batches()

    elif task.lower() in [
        "create_unique_batch",
        "cub",
        "unique_batch",
        "batch",
        "unique",
        "one_batch",
    ]:
        click.echo(f"Prepare to unique batch creation.")
        create_unique_batch()

    elif task.lower() in ["index", "indexing", "make_indices", "i"]:
        click.echo(f"Prepare to indexing.")
        index()

    elif task.lower() in ["check_report", "show_report", "report", "rep", "r"]:
        click.echo(f"Prepare to show report")
        check_report()

    else:
        print(
            'Unknown query. Only "search", "create_batches", "create_unique_batch", "index" tasks are available.'
        )


if __name__ == "__main__":
    start_app()
