import os
import pathlib

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display

import click

BASE_PATH = pathlib.Path(__file__).parent.resolve()


@click.command()
@click.option("--epochs", "-e", default=10)
@click.option("--n-gpus", default=1)
@click.option("--learning-rate", "-l", type=float, default=3e-5)
@click.argument("task_name")
def cli(task_name, epochs, n_gpus, learning_rate):
    """Run a single-task experiment"""
    # Tokenize and cache task
    tokenize_and_cache.main(
        tokenize_and_cache.RunConfiguration(
            task_config_path=f"workdata/tasks/configs/{task_name}_config.json",
            hf_pretrained_model_name_or_path="roberta-base",
            output_dir=f"workdata/cache/{task_name}",
            phases=["train", "val", "test"],
            max_seq_length=128,
            smart_truncate=True,
        )
    )

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="workdata/tasks/configs",
        task_cache_base_path="workdata/cache",
        train_task_name_list=[task_name],
        val_task_name_list=[task_name],
        test_task_name_list=[task_name],
        train_batch_size=16,
        eval_batch_size=8,
        epochs=epochs,
        num_gpus=n_gpus,
    ).create_config()

    os.makedirs("workdata/run_configs/", exist_ok=True)
    py_io.write_json(
        jiant_run_config, f"workdata/run_configs/{task_name}.json"
    )
    display.show_json(jiant_run_config)

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=str(BASE_PATH / f"workdata/run_configs/{task_name}.json"),
        output_dir=str(BASE_PATH / f"workdata/runs/run-single-{task_name.replace('_', '-')}/{learning_rate}"),
        hf_pretrained_model_name_or_path="roberta-base",
        model_path=str(BASE_PATH / "workdata/models/roberta-base/model/model.p"),
        model_config_path=str(BASE_PATH / "workdata/models/roberta-base/model/config.json"),
        learning_rate=learning_rate,
        eval_every_steps=100,
        do_train=True,
        do_val=True,
        do_save_best=True,
        write_val_preds=True,
        write_test_preds=True,
        force_overwrite=True,
        seed=6247359,
    )
    main_runscript.run_loop(run_args)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
