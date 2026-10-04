"""Exercise baseline loading/generation before the full pilot, without scoring."""
import argparse
from pathlib import Path
from mscd.types import ModelArtifact, Request, GenerationConfig
from mscd.decoding.generators import (
    ModelGenerator,
    MergedLoRAGenerator,
    WholeOutputConsensusGenerator,
)
from mscd.artifacts import tree_identity, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--method", choices=["single", "merge", "whole"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    teachers = [
        ModelArtifact(
            str(Path(args.references) / name),
            args.base,
            args.base,
            "teacher",
            tree_identity(Path(args.references) / name),
        )
        for name in ["pi_A_retrained", "pi_B_retrained"]
    ]
    factory = {
        "single": lambda: ModelGenerator(teachers[0]),
        "merge": lambda: MergedLoRAGenerator(teachers),
        "whole": lambda: WholeOutputConsensusGenerator(teachers),
    }
    generator = factory[args.method]()
    responses = list(
        generator.generate(
            [Request("fixture:0", "Explain photosynthesis simply.", 0)],
            GenerationConfig(max_new_tokens=16, max_attempts=2),
        )
    )
    assert len(responses) == 1 and responses[0].status in {"completed", "abstained"}
    if args.method != "whole":
        assert responses[0].status == "completed" and responses[0].response
    atomic_json(
        args.output,
        dict(
            passed=True,
            fixture_only=True,
            method=args.method,
            status=responses[0].status,
        ),
    )


if __name__ == "__main__":
    main()
