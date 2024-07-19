from dataclasses import dataclass, field
import time

from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

from .analyze import AnalysisResults, BinType
from ...configs import GeneralConfig, ResourcesConfig
from ....io import load_dataclass_jsonl


@dataclass
class Config:
    # How many columns to have in the figure before soft wrapping.
    cols_wrap: int = 2

    # How much to scale the results. E.g., 100.0 to scale from proportion to percentage.
    scale: float = 100.0

    # The number of answer choices in a QAData's answer_choices.
    num_choices: int = 2

    # The height of the resulting figure.
    height: int = 500  # Use 500 for 3 plots/LLMs. Use 1000 for 7 plots/LLMs.

    # The spacing between each row of sub-plots.
    vertical_spacing: float = 0.07  # Use 0.07 for 3 plots. Use 0.03 for 7 plots.

    # Which LLMs to include in the figure.
    llms: dict[str, str] = field(default_factory=dict)


class Command:
    """Generates Plotly figures from graph analysis data."""

    FACTUAL = "Factual"
    ANTI_FACTUAL = "Anti-Factual"

    def __init__(self, resources: ResourcesConfig, _: GeneralConfig, cfg: Config):
        self.resources = resources
        self.cfg = cfg
        self.x, self.marginal = "bin_value", "Tree Size"
        self.y, self.llm, self.series = "y", "llm", "series"
        self.data, self.baselines = None, None

    def run(self):
        results, file_path = {}, self.resources.graph_analysis_data_file
        for result in load_dataclass_jsonl(file_path, AnalysisResults):
            results.setdefault(result.bin_type, []).append(result)
        for bin_type, analyses in results.items():
            df = self._update_data_and_baseline(analyses)
            fig = self._create_figure(bin_type)
            self._add_traces(fig, df)
            self._add_legend(fig)
            for (n, b), s in zip(self.baselines.items(), ["dash", "dot"]):
                self._add_baseline(fig, n, b, s)
            self._update_axes(fig)
            self._save_figure(fig, bin_type)

    def _update_data_and_baseline(self, results: list[AnalysisResults]) -> pd.DataFrame:
        self.data = {
            self.x: [],
            self.y: [],
            self.marginal: [],
            self.llm: [],
            self.series: [],
        }
        self.baselines = {"No Context": {}, "Random": {}}
        for result in results:
            if result.llm not in self.cfg.llms:
                continue
            llm_value = self.cfg.llms[result.llm]
            if result.is_baseline:
                baseline_value = result.f_acc * self.cfg.scale
                random_value = self.cfg.scale / self.cfg.num_choices
                self.baselines["No Context"].setdefault(llm_value, baseline_value)
                self.baselines["Random"].setdefault(llm_value, random_value)
            else:
                for series_value in [self.FACTUAL, self.ANTI_FACTUAL]:
                    self.data[self.x].append(result.bin_value)
                    self.data[self.marginal].append(result.tree_size)
                    y_ = result.f_acc if series_value == self.FACTUAL else result.af_acc
                    # se_ = result.f_se if series_value == self.FACTUAL else result.af_se
                    self.data[self.y].append(y_ * self.cfg.scale)
                    # data[se].append(se_ * self.cfg.scale)
                    self.data[self.llm].append(llm_value)
                    self.data[self.series].append(series_value)
        return pd.DataFrame(self.data)

    def _row_col(self, zero_based_count: int) -> tuple[int, int]:
        n_cols = self.cfg.cols_wrap
        return 1 + zero_based_count // n_cols, 1 + zero_based_count % n_cols

    def _llms(self) -> list[str]:
        return sorted(set(self.data[self.llm]))

    def _marginal(self) -> list[int]:
        return sorted(set(self.data[self.marginal]))

    def _create_figure(self, bin_type: BinType) -> go.Figure:
        fig = make_subplots(
            rows=self._row_col(len(self._llms()) - 1)[0],
            cols=self.cfg.cols_wrap,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=self.cfg.vertical_spacing,
            row_titles=self._llms(),
            column_titles=[bin_type.value.replace("_", " ").title()]
        )
        fig.update_layout(
            template="simple_white",
            margin=dict(l=60, r=180, t=35, b=0),
            font=dict(size=12),
            width=375,
            height=self.cfg.height,
        )
        fig.for_each_annotation(
            lambda a: a.update(
                text=f"<b>{a.text}</b>" if a.text in self._llms() else a.text,
                font=dict(size=16),
                xshift=15 if a.text in self._llms() else a.yshift,
            ),
        )
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
        return fig

    def _add_traces(self, fig: go.Figure, df: pd.DataFrame):
        for llm_count, llm in enumerate(self._llms()):
            llm_df = df[df[self.llm] == llm]
            row, col = self._row_col(llm_count)
            for series, props in self._series_props(fig).items():
                series_df = llm_df[llm_df[self.series] == series]
                for marginal_count, marginal in enumerate(self._marginal()):
                    marginal_df = series_df[series_df[self.marginal] == marginal]
                    fig.add_trace(
                        go.Scatter(
                            x=marginal_df[self.x],
                            y=marginal_df[self.y],
                            marker=dict(
                                color=props["color"][marginal_count],
                                symbol=props["symbol"],
                            ),
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

    def _add_legend(self, fig: go.Figure):
        fig.update_layout(
            legend=dict(
                font=dict(size=14),
                grouptitlefont=dict(size=15),
                tracegroupgap=5,
                yanchor="top",
                xanchor="right",
                xref="container",
                x=1.0,
            ),
        )

        x, y = [min(self.data[self.x])], [-max(self.data[self.y])]
        for series, props in self._series_props(fig).items():
            for marginal_count, marginal in enumerate(self._marginal()):
                legend_group_title_text = (
                    f'{self.marginal}<br>  (<span style="color: {props["bar_color"]}; '
                    f'font-style:italic;">{series}</span>)'
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=marginal,
                        mode="markers",
                        marker=dict(
                            color=props["color"][marginal_count],
                            symbol=props["symbol"],
                        ),
                        legendgroup=series,
                        legendgrouptitle_text=legend_group_title_text,  # noqa
                    ),
                    row=1,
                    col=1,
                )

    def _add_baseline(self, fig: go.Figure, name: str, baseline: dict, dash_style: str):
        for count, llm in enumerate(self._llms()):
            row, col = self._row_col(count)
            fig.add_hline(
                row=row,
                col=col,
                line_width=3,
                opacity=0.25,
                line_dash=dash_style,
                y=baseline[llm],
            )
        fig.add_trace(
            go.Scatter(
                x=[1, 1],
                y=[0, 0],
                name=name,
                mode="lines",
                line=dict(dash=dash_style, width=3, color="rgba(0, 0, 0, 0.25)"),
                legendgroup="baselines",
                legendgrouptitle_text="Baselines",  # noqa
            ),
            row=1,
            col=1,
        )

    def _update_axes(self, fig: go.Figure):
        fig.update_yaxes(
            range=[0, self.cfg.scale * 1.05],
            tickvals=[0, self.cfg.scale / 2, self.cfg.scale],
            showticklabels=True,
        )
        fig.for_each_xaxis(lambda x_axis: x_axis.update(showticklabels=True, dtick=1))

    def _save_figure(self, fig: go.Figure, bin_type: BinType):
        file_path = self.resources.graph_analysis_pretty_file
        file_path = file_path.format(bin_type.value)
        if file_path.endswith(".png"):
            fig.write_image(file_path, scale=3.0)
        else:
            # Write PDF image twice with a delay to load MathJax. See:
            #  https://github.com/plotly/plotly.py/issues/3469
            fig.write_image(file_path)
            time.sleep(1)
            fig.write_image(file_path)

    def _series_props(self, fig: go.Figure) -> dict:
        f = sample_colorscale("gnbu", len(self._marginal()), low=0.25, high=0.75)
        af = sample_colorscale("ylorrd", len(self._marginal()), low=0.2, high=0.8)
        b_color = fig.layout.template.layout.colorway
        return {
            self.FACTUAL: {"color": f, "symbol": "circle", "bar_color": b_color[0]},
            self.ANTI_FACTUAL: {"color": af, "symbol": "x", "bar_color": b_color[1]},
        }
