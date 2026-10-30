"""Command-line interface for Agentic DermaGuard."""

import argparse
import json
import sys
from pathlib import Path

from src.agents.image_quality_agent import ImageQualityAgent
from src.agents.diagnosis_agent import DiagnosisAgent
from src.config import MODELS_DIR, REPORTS_DIR
from src.pipeline.train import train_diagnosis_model
from src.pipeline.validator import run_validation_pipeline
from src.triage.triage_engine import TriageEngine


def main():
    parser = argparse.ArgumentParser(
        prog="dermaguard",
        description="Agentic DermaGuard: Multi-Agent Skin Lesion Screening & Triage CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command 1: assess image quality
    parser_quality = subparsers.add_parser("assess-quality", help="Assess a single dermoscopic image for quality defects")
    parser_quality.add_argument("--image", type=str, required=True, help="Path to image file")

    # Command 2: full single-image triage
    parser_triage = subparsers.add_parser("triage", help="Run full multi-agent triage on an image")
    parser_triage.add_argument("--image", type=str, required=True, help="Path to image file")
    parser_triage.add_argument("--checkpoint", type=str, default=None, help="Optional trained model checkpoint")

    # Command 3: train diagnosis model
    parser_train = subparsers.add_parser("train", help="Train the diagnosis model on HAM10000 images")
    parser_train.add_argument("--images", type=str, required=True, help="Directory containing training images")
    parser_train.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV (image_id, dx)")
    parser_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser_train.add_argument("--output", type=str, default=str(MODELS_DIR / "diagnosis_model.pth"), help="Output checkpoint path")

    # Command 4: validate cohort (Day 1 / Level 1 Hackathon task)
    parser_val = subparsers.add_parser("validate", help="Run multi-agent validation over the 2,000 validation images")
    parser_val.add_argument("--val-dir", type=str, required=True, help="Directory containing 2,000 validation images")
    parser_val.add_argument("--ground-truth", type=str, default=None, help="Path to validation ground-truth CSV")
    parser_val.add_argument("--checkpoint", type=str, default=None, help="Optional trained model checkpoint")
    parser_val.add_argument("--output-json", type=str, default=str(REPORTS_DIR / "validation_submission_payload.json"), help="Output JSON payload path")
    parser_val.add_argument("--limit", type=int, default=None, help="Optional limit for rapid verification")

    args = parser.parse_args()

    if args.command == "assess-quality":
        agent = ImageQualityAgent()
        res = agent.assess(args.image)
        print(json.dumps(res.model_dump(), indent=2))

    elif args.command == "triage":
        quality_agent = ImageQualityAgent()
        diagnosis_agent = DiagnosisAgent(model_path=args.checkpoint)
        engine = TriageEngine()

        q_res = quality_agent.assess(args.image)
        d_res = diagnosis_agent.predict(args.image) if q_res.quality_status != "FAIL" else None
        t_res = engine.evaluate(image_id=Path(args.image).stem, quality=q_res, diagnosis=d_res)
        print(json.dumps(t_res.model_dump(), indent=2))

    elif args.command == "train":
        train_diagnosis_model(
            train_image_dir=Path(args.images),
            metadata_csv_path=Path(args.metadata),
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_checkpoint=Path(args.output),
        )

    elif args.command == "validate":
        run_validation_pipeline(
            val_image_dir=Path(args.val_dir),
            val_ground_truth_csv=Path(args.ground_truth) if args.ground_truth else None,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            save_json_path=Path(args.output_json),
            limit=args.limit,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

