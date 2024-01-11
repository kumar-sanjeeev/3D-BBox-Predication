import hydra

from pointfusion.datasets.process.process_dl_data import ProcessData


@hydra.main(version_base=None, config_path="../configs", config_name="preprocess")
def main(cfg):
    pp = ProcessData(cfg.root_path, cfg.window_size, cfg.output_dir_path)


if __name__ == "__main__":
    main()
