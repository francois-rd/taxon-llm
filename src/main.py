from collections import namedtuple
from typing import Any
import random

import coma

from tlm.commands import analyze, prompt, configs as cfgs, loaders
from tlm.components import SimpleLLMOutputParser
from tlm.llms import (
    DummyConfig,
    DummyLLM,
    OpenAIConfig,
    OpenAILLM,
    TransformersConfig,
    TransformersLLM,
)


# Make named tuples out of each config type.
ConfigData = namedtuple("ConfigData", "id_ type_")
analysis_loader = ConfigData("analysis_loader", cfgs.AnalysisLoaderConfig)
general = ConfigData("general", cfgs.GeneralConfig)
resources = ConfigData("resources", cfgs.ResourcesConfig)
surfacer = ConfigData("surfacer", cfgs.InstanceSurfacerConfig)

dummy = ConfigData("dummy", DummyConfig)
openai = ConfigData("openai", OpenAIConfig)
transformers = ConfigData("transformers", TransformersConfig)

basic_analysis = ConfigData("basic_analysis", analyze.basic.analyze.Config)
basic_pretty = ConfigData("basic_pretty", analyze.basic.pretty.Config)
graph_analysis = ConfigData("graph_analysis", analyze.graph.analyze.Config)
graph_pretty = ConfigData("graph_pretty", analyze.graph.pretty.Config)


def as_dict(*cfgs_data: ConfigData):
    """Converts the given config data to a valid coma config dict."""
    return {cfg.id_: cfg.type_ for cfg in cfgs_data}


@coma.hooks.hook
def prompt_csqa_small(name: str, configs: dict[str, Any]):
    # Grab the LLM info.
    if "dummy" in name:
        llm_cfg_id, llm_class = dummy.id_, DummyLLM
    elif "openai" in name:
        llm_cfg_id, llm_class = openai.id_, OpenAILLM
    elif "transformers" in name:
        llm_cfg_id, llm_class = transformers.id_, TransformersLLM
    else:
        raise ValueError(f"Unsupported prompt command: {name}")

    # Grab the initialized configs.
    general_cfg: cfgs.GeneralConfig = configs[general.id_]
    srcs_cfg: cfgs.ResourcesConfig = configs[resources.id_]
    llm_cfg = configs.pop(llm_cfg_id)

    # Set the random seed as early as possible.
    random.seed(general_cfg.random_seed)

    # Append the pseudo config to the (ordered) configs dictionary.
    configs["pseudo_cfg"] = prompt.PseudoConfig(
        dataset=loaders.dataset_loader(srcs_cfg, general_cfg),
        llm=llm_class(srcs_cfg.llm, llm_cfg),
    )


@coma.hooks.hook
def analyze_csqa_small(name: str, configs: dict[str, Any]) -> Any:
    # Grab the analysis type info.
    if "basic" in name:
        pseudo_cfg = analyze.basic.analyze.PseudoConfig
    elif "graph" in name:
        pseudo_cfg = analyze.graph.analyze.PseudoConfig
    else:
        raise ValueError(f"Unsupported analysis command: {name}")

    # Grab the initialized configs.
    srcs_cfg: cfgs.ResourcesConfig = configs[resources.id_]
    loader_cfg: cfgs.AnalysisLoaderConfig = configs.pop(analysis_loader.id_)

    # Append the pseudo config to the (ordered) configs dictionary.
    configs["pseudo_cfg"] = pseudo_cfg(
        analyses=loaders.analysis_loader(
            resources=srcs_cfg,
            loader_cfg=loader_cfg,
            qa_dataset=loaders.csqa_loader(srcs_cfg),
            llm_parser=SimpleLLMOutputParser(),
        ),
    )


@coma.hooks.hook
def pre_run_hook(known_args):
    """This pre-run hook exists early. Useful for debugging init hooks."""
    if known_args.dry_run:
        print("Dry run.")
        quit()


if __name__ == "__main__":
    # Initialize.
    dry_run_hook = coma.hooks.parser_hook.factory(
        "--dry-run",
        action="store_true",
        help="exit during pre-run",
    )
    coma.initiate(
        parser_hook=coma.hooks.sequence(coma.hooks.parser_hook.default, dry_run_hook),
        post_config_hook=coma.hooks.post_config_hook.multi_cli_override_factory(
            coma.config.cli.override_factory(sep="::"),
        ),
        pre_run_hook=pre_run_hook,
        **as_dict(resources, general),
    )

    # Prompt commands.
    coma.register(
        "prompt.dummy.csqa.small",
        prompt.Command,
        pre_init_hook=prompt_csqa_small,
        **as_dict(surfacer, dummy),
    )
    coma.register(
        "prompt.openai.csqa.small",
        prompt.Command,
        pre_init_hook=prompt_csqa_small,
        **as_dict(surfacer, openai),
    )
    coma.register(
        "prompt.transformers.csqa.small",
        prompt.Command,
        pre_init_hook=prompt_csqa_small,
        **as_dict(surfacer, transformers),
    )

    # Analysis commands.
    coma.register(
        "analyze.basic.csqa.small",
        analyze.basic.analyze.Command,
        pre_init_hook=analyze_csqa_small,
        **as_dict(basic_analysis, analysis_loader),
    )
    coma.register(
        "analyze.basic.pretty",
        analyze.basic.pretty.Command,
        **as_dict(basic_pretty),
    )
    coma.register(
        "analyze.graph.csqa.small",
        analyze.graph.analyze.Command,
        pre_init_hook=analyze_csqa_small,
        **as_dict(graph_analysis, analysis_loader),
    )
    coma.register(
        "analyze.graph.pretty",
        analyze.graph.pretty.Command,
        **as_dict(graph_pretty),
    )

    # Run.
    coma.wake()
