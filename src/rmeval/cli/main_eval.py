import os
import asyncio
from datetime import datetime
from pprint import pprint

import hydra
import omegaconf
from loguru import logger
from omegaconf import OmegaConf

from rmeval import TaskRunner, make_rm


@hydra.main(config_path="config", config_name="main_eval", version_base=None)
def main(config: omegaconf.DictConfig):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    OmegaConf.resolve(config)
    pprint(OmegaConf.to_container(config))

    model_id = OmegaConf.select(config, "model.id")
    if not model_id:
        model_path = OmegaConf.select(config, "model.path")
        if not model_path:
            raise ValueError("`model.path` must be provided in config or via CLI.")
        model_id = os.path.basename(str(model_path).rstrip("/"))

    exp_dir = os.path.join(
        config.output.root_dir,
        model_id,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    config.output.exp_dir = exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(exp_dir, "args.yaml"))

    logger.add(
        os.path.join(exp_dir, "result.log"),
        level="INFO",
        enqueue=True,
        encoding="utf-8",
    )
    task_runner = TaskRunner()
    rm = make_rm(config.rm.name)
    task_runner.run(config, logger, rm)


if __name__ == "__main__":
    main()
