from dataclasses import dataclass, field
import time

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .analyze import AnalysisResults, BinType
from ...configs import GeneralConfig, ResourcesConfig
from ....base import RelationType
from ....io import load_dataclass_jsonl


@dataclass
class Config:
    # How much to scale the results. E.g., 100.0 to scale from proportion to percentage.
    scale: float = 100.0

    # The number of answer choices in a QAData's answer_choices.
    num_choices: int = 2

    # The height of the resulting figure.
    height: int = 500  # Use 500 for 3 plots/LLMs. Use 1000 for 7 plots/LLMs.

    # The spacing between each row of sub-plots.
    facet_row_spacing: float = 0.07  # Use 0.07 for 3 plots. Use 0.03 for 7 plots.

    # Swap series vs facet_col from F/AF and skill_type to the converse.
    swap_series_and_cols: bool = False

    # Which LLMs to include in the figure.
    llms: dict[str, str] = field(default_factory=dict)

    # Which reasoning skill types to include in the figure, in display order.
    skill_types: dict[RelationType, str] = field(default_factory=dict)


class Command:
    """Generates Plotly figures from BasicAnalysis data."""

    FACTUAL = "Factual"
    ANTI_FACTUAL = "Anti-Factual"

    def __init__(self, resources: ResourcesConfig, _: GeneralConfig, cfg: Config):
        self.resources = resources
        self.cfg = cfg

    def run(self):
        results, file_path = {}, self.resources.basic_analysis_data_file
        for result in load_dataclass_jsonl(file_path, AnalysisResults):
            results.setdefault(result.bin_type, []).append(result)
        for bin_type, analyses in results.items():
            self._plot(bin_type, analyses)

    def _plot(self, bin_type: BinType, results: list[AnalysisResults]):
        x, y, se, llm, skill_type, series = "x", "y", "se", "llm", "skilltype", "series"
        data = {x: [], y: [], se: [], llm: [], skill_type: [], series: []}
        baselines, randoms, all_skill_types = {}, {}, []
        for result in results:
            if result.llm not in self.cfg.llms:
                continue
            if result.skill_type not in self.cfg.skill_types:
                continue
            llm_value = self.cfg.llms[result.llm]
            skill_type_value = self.cfg.skill_types[result.skill_type]
            if result.is_baseline:
                baseline_value = result.f_acc * self.cfg.scale
                random_value = self.cfg.scale / self.cfg.num_choices
                baselines[llm_value] = baseline_value
                randoms[llm_value] = random_value
            else:
                for series_value in [self.FACTUAL, self.ANTI_FACTUAL]:
                    data[x].append(result.skill_count)
                    y_ = result.f_acc if series_value == self.FACTUAL else result.af_acc
                    se_ = result.f_se if series_value == self.FACTUAL else result.af_se
                    data[y].append(y_ * self.cfg.scale)
                    data[se].append(se_ * self.cfg.scale)
                    data[llm].append(llm_value)
                    data[skill_type].append(skill_type_value)
                    data[series].append(series_value)
        df = pd.DataFrame(data)

        fig = px.bar(
            df,
            x=x,
            y=y,
            error_y=se,
            color=skill_type if self.cfg.swap_series_and_cols else series,
            barmode="group",
            facet_row=llm,
            facet_col=series if self.cfg.swap_series_and_cols else skill_type,
            facet_row_spacing=self.cfg.facet_row_spacing,  # default is 0.07
            facet_col_spacing=0.02,  # default is 0.03
            category_orders={
                x: sorted(set(data[x])),
                llm: sorted(self.cfg.llms.values()),
                skill_type: self.cfg.skill_types.values(),
            },
            template="simple_white",
            width=1000 if self.cfg.swap_series_and_cols else 1000,  # default is 700
            height=self.cfg.height,  # default is 500
        )

        self.add_baselines(fig, baselines, "Baseline", "dash")
        self.add_baselines(fig, randoms, "Random", "dot")

        # Replace col names: "skill_type=value" -> "value"
        # Replace row names: "llm=value" -> "<b>value</b>"
        fig.for_each_annotation(
            lambda a: a.update(
                text=(
                    a.text.replace(f"{series}=", "").replace(f"{skill_type}=", "")
                    if series in a.text or skill_type in a.text
                    else f"<b>{a.text.replace(f'{llm}=', '')}</b>"
                ),
                font=(
                    dict(size=14 if llm in a.text else 16)
                    if llm in a.text or series in a.text
                    else dict(size=16, family="Courier New, monospace")
                ),
                xshift=15 if llm in a.text else a.xshift,
                yshift=10 if (series in a.text or skill_type in a.text) else a.yshift,
            ),
        )

        fig.update_yaxes(
            title=None,
            range=[0, self.cfg.scale * 1.05],
            tickvals=[0, self.cfg.scale / 2, self.cfg.scale],
        )
        fig.update_xaxes(title=None, dtick=1)
        fig.for_each_xaxis(lambda x_axis: x_axis.update(showticklabels=True))
        fig.add_annotation(
            showarrow=False,
            text=f"Accuracy (%)",
            textangle=-90,
            x=0,
            xanchor="center",
            xref="paper",
            y=0.5,
            yanchor="middle",
            yref="paper",
            xshift=-50,
            font=dict(size=16),
        )

        fig.update_layout(
            legend=dict(  # Legend orientation and position.
                title=None,
                orientation="h",
                y=-0.15,  # Smarter way to position outside plot without negative???
                yanchor="bottom",
                x=0.5,
                xanchor="center",
                visible=True,
                font=dict(size=14 if self.cfg.swap_series_and_cols else 16),
            ),
            margin=dict(l=60, r=20, t=35, b=0),
        )

        # Update the look of the error bars.
        fig.update_traces(error_y=dict(thickness=1, width=3, color="black"))

        # Save plot to file.
        file_path = self.resources.basic_analysis_pretty_file
        file_path = file_path.format(self.cfg.swap_series_and_cols, bin_type.value)
        if file_path.endswith(".png"):
            fig.write_image(file_path, scale=3.0)
        else:
            # Write PDF image twice with a delay to load MathJax. See:
            #  https://github.com/plotly/plotly.py/issues/3469
            fig.write_image(file_path)
            time.sleep(1)
            fig.write_image(file_path)

    def add_baselines(self, fig, baselines, name, dash_style):
        # Add a horizontal baseline specific to each subplot.
        num_cols = 2 if self.cfg.swap_series_and_cols else len(self.cfg.skill_types)
        for row, llm in enumerate(self._reversed_llms()):
            for col in range(num_cols):
                fig.add_hline(
                    row=row + 1,
                    col=col + 1,
                    line_width=3,
                    opacity=0.25,
                    line_dash=dash_style,
                    y=baselines[llm],
                )

        # Create a trace to add to the legend. Should draw nothing since we are in
        # mode='lines' and the data collapses to a single point.
        trace = go.Scatter(
            x=[1, 1],
            y=[0, 0],
            name=name,
            mode="lines",
            line=dict(dash=dash_style, width=3, color="rgba(0, 0, 0, 0.25)"),
        )
        trace.update(legendgroup=name, showlegend=True)

        # Add it to just one facet so that it appears only once in the legend.
        fig.add_trace(trace, row=1, col=1)

    def _reversed_llms(self):
        return reversed(sorted(self.cfg.llms.values()))
