# Paper Matrix

| Key | Bucket | Model family | Dataset/task | Signal | Intervention | Claim type | Threat to this project | How to cite/distinguish |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| burns2022ccs | hidden truth | GPT-style LMs | truth probes | contrast-consistent activations | none | diagnostic | Latent probes may recover broad truth/difficulty, not trajectory correctness | Cite as latent-knowledge precedent; distinguish matched rollout and control use |
| azaria2023internal | hidden truth | transformer LMs | truthfulness statements | hidden-state classifier | none | diagnostic | "Knows when lying" language can be overclaimed | Cite as truth-detection baseline; avoid anthropomorphic claims |
| kadavath2022language | hidden truth | LM families | calibration/truthfulness | verbal and internal confidence | selective prediction | diagnostic/control-adjacent | Calibration baselines may match hidden probes | Include verbal confidence and calibration controls |
| wei2022cot | trajectories | PaLM/GPT-style | arithmetic/reasoning | prompted CoT text | prompting | control | CoT itself is a stronger baseline than direct answer | Treat as base prompting setup |
| wang2022selfconsistency | trajectories/control | GPT-style | math/commonsense | answer agreement | sample-and-vote | control | Self-consistency may dominate hidden-triggered extra compute | Use same-token-budget fixed and selective SC |
| zhou2022leasttomost | trajectories | GPT-3 | compositional reasoning | decomposed prompts | prompting | control | Alternate strategies may help without hidden probes | Include alternate strategy prompt baseline |
| madaan2023selfrefine | self-correction | GPT-style | generation/reasoning | verbal critique | iterative refinement | control | Always-refine may match selective reassess | Compare always vs hidden-triggered reassess |
| shinn2023reflexion | self-correction | GPT-style agents | decision/reasoning | verbal reflection | restart/memory | control | Restart policies may not require hidden states | Include random and always restart |
| turner2023actadd | steering | GPT-2/J models | behavior steering | activation difference | activation addition | causal/control | Steering can change behavior but not necessarily correctness | Use as steering method, not proof of correctness causality |
| zou2023representation | steering | open LMs | honesty/toxicity/etc. | representation direction | representation engineering | causal/control | Directions may steer style/safety rather than math correctness | Include random/opposite/shuffled controls |
| todd2024functionvectors | steering/causal | transformer LMs | in-context learning | function vectors | activation patching/addition | causal | Hidden directions can encode task functions | Distinguish task-vector control from correctness recovery |
| openai2026monitorability | causal/latent reasoning | reasoning models | CoT monitorability evals | visible CoT monitor signals | monitoring | diagnostic/control-adjacent | CoT text may be less monitorable as models internalize reasoning | Use to motivate hidden-state and text-monitor comparison |
| anthropic2025faithfulness | causal/latent reasoning | Claude reasoning models | faithfulness tests | CoT faithfulness | none | diagnostic | Visible CoT may omit causal factors | Justifies partial-CoT text baseline but cautions against relying on it |
| deepconf2025 | inference control | open reasoning models | AIME/GPQA/math | trace confidence | filter/stop traces | control | Strong direct competitor for token-efficient routing | Must compare to DeepConf-style trace confidence |
| hiddenerror2026 | CoT hidden correctness | Qwen/Llama/Phi/DeepSeek-style | reasoning traces | hidden-state probe | steering/ablation stress tests | diagnostic vs causal | Directly overlaps early correctness signal and negative causal result | Cite as closest SOTA; distinguish by matched problem and policies |
| cotfails2026 | CoT hidden correctness | reasoning LMs | failed CoT tasks | hidden states and token patches | activation patching | diagnostic/causal | Hidden states may contain recoverable solution information | Include activation-patching/steering controls and parse/coherence metrics |

## Matrix Fields To Fill In Zotero

For each added paper, record citation key, URL/DOI, model family, dataset,
signal type, intervention type, diagnostic/causal/control claim, threatened
baseline, and the sentence-level role in this project.
