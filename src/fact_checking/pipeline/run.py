from __future__ import annotations

import hydra
from omegaconf import DictConfig

from fact_checking.pipeline.runner import PipelineRunner


@hydra.main(version_base=None, config_path="../../../configs", config_name="pipeline/default")
def main(cfg: DictConfig) -> None:
    manifest = PipelineRunner(cfg).run()
    print(f"Pipeline completed: {manifest['run_dir']}")


if __name__ == "__main__":
    main()
