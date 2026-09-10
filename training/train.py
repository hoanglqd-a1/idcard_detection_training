import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / 'yolo_training_report'
MODEL_PATH = ROOT / 'model' / 'yolov8s-detect.pt'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train YOLO OBB on a prepared dataset.')
    parser.add_argument(
        '--data',
        type=Path,
        default=ROOT / 'datasets' / 'MIDV500' / 'data.yaml',
        help='Path to the dataset YAML.',
    )
    args = parser.parse_args(argv)
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f'Dataset YAML not found: {data_path}. Run download_dataset.py first or pass --data.')

    from ultralytics import YOLO

    model = YOLO(str(REPORT_DIR / 'yolov8s-obb.pt'))
    model.train(
        data=str(data_path),
        epochs=100,
        batch=16,
        imgsz=640,
        patience=10,
        degrees=30,
        flipud=0.5,
        fliplr=0.5,
        lr0=1e-2,
        device=0,
        workers=2,
        project=str(ROOT),
        name=REPORT_DIR.name,
        exist_ok=True,
        save=True,
    )

    best_path = Path(model.trainer.best)
    if not best_path.is_file():
        raise FileNotFoundError(f'Training did not produce a best checkpoint: {best_path}')
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, MODEL_PATH)
    print(f'Training report: {model.trainer.save_dir}')
    print(f'Best model copied to: {MODEL_PATH}')


if __name__ == '__main__':
    main()
