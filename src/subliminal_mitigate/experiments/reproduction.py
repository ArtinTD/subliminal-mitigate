"""CPU-only reproduction from outcome-only public resources."""
from importlib import resources
import json
from pathlib import Path

from ..evaluation.ratio import MassiveMedicalRatioAnalysis
from ..evaluation.reports import write_reports


def load_resource(name):
    if name not in {"massive_observations.json", "massive_constituents.json", "em_outcomes.json"}:
        raise ValueError("unknown bundled outcome resource")
    return json.loads(resources.files("subliminal_mitigate").joinpath("resources", name).read_text())


class PaperReproduction:
    """One reviewable entry point for both final result sets, without paid calls."""

    def __init__(self, experiment="all", *, plots=True):
        if experiment not in {"all", "em", "massive"}:
            raise ValueError("experiment must be em, massive, or all")
        self.experiment = experiment
        self.plots = plots

    def run(self, output_dir):
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError("choose a new output directory; previous artifacts are not overwritten")
        output_dir.mkdir(parents=True)
        artifacts = {}
        if self.experiment in {"all", "massive"}:
            observations = load_resource("massive_observations.json")
            result = MassiveMedicalRatioAnalysis().analyze(observations)
            destination = output_dir / "massive"
            write_reports(result, destination)
            artifacts["massive_arms"] = len(result["arms"])
            if self.plots:
                from ..plotting.ratio import RatioFigure
                artifacts["massive_figure"] = RatioFigure(
                    result, load_resource("massive_constituents.json")
                ).render(destination)["outputs"]
        if self.experiment in {"all", "em"}:
            outcomes = load_resource("em_outcomes.json")
            destination = output_dir / "em"
            destination.mkdir(parents=True)
            (destination / "results.json").write_text(json.dumps(outcomes, indent=2) + "\n")
            artifacts["em_methods"] = len(outcomes["rows"])
            if self.plots:
                from ..plotting.em import EMFigure
                figure = EMFigure(outcomes["rows"])
                artifacts["em_figure"] = figure.render(destination)
                artifacts["em_complete_figure"] = figure.render(destination, complete_kalai=True)
        result = {"experiment": self.experiment, "artifacts": artifacts,
                  "gpu_jobs": 0, "external_api_calls": 0,
                  "output_dir": str(output_dir.resolve())}
        (output_dir / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
