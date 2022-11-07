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
@click.argument("task_names", nargs=-1)
@click.option("--epochs", "-e", default=10)
@click.option("--learning-rate", "-l", type=float, default=3e-5)
@click.option("--n-gpus", default=1)
def cli(task_names, epochs, learning_rate, n_gpus):
    task_names = list(task_names)  # needs to be a list or jiant will complain
    reg = any("_reg" in task for task in task_names)

    # Tokenize and cache each task
    for task_name in task_names:
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

    # Write config for the multi-head experiment emo+apppraisals
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="workdata/tasks/configs",
        task_cache_base_path="workdata/cache",
        train_task_name_list=task_names,
        val_task_name_list=task_names,
        test_task_name_list=task_names,
        train_batch_size=16,
        eval_batch_size=8,
        epochs=epochs,
        num_gpus=n_gpus,
    ).create_config()

    os.makedirs("workdata/run_configs/", exist_ok=True)
    task_names_id = f"multi-{'reg' if reg else 'cls'}-{'with' if 'emo_cls' in task_names else 'without'}"
    py_io.write_json(jiant_run_config, f"workdata/run_configs/run_{task_names_id}.json")
    display.show_json(jiant_run_config)

    # Train multi-head

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=str(BASE_PATH / f"workdata/run_configs/run_{task_names_id}.json"),
        output_dir=str(BASE_PATH / f"workdata/runs/run-{task_names_id}/{learning_rate}"),
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
