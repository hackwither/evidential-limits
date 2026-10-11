// The eight evaluation suites audited in Kaur (2026), §5 and Figure 2, with the sample sizes the
// paper uses (the reproducibility code's analysis/power_analysis.py holds the same table).
// The paper treats n as the only suite-specific input; harm rates are assumed, common across suites.
//
// `id` is frozen once shipped: /evidential-ceiling/b/<id>/ is a citable permalink.
// To add a suite from a later audit, append an object with a sourced `n` and `source`.

export const SUITES = [
  { id: "advbench", name: "AdvBench", n: 520, cite: "Zou et al. 2023", source: "§5.1, Fig. 2", note: "" },
  { id: "harmbench", name: "HarmBench", n: 400, cite: "Mazeika et al. 2024", source: "§5.1, Fig. 2", note: "" },
  { id: "strongreject", name: "StrongREJECT", n: 313, cite: "Souly et al. 2024", source: "§5.2, Fig. 2", note: "" },
  { id: "wildguard", name: "WildGuard", n: 1725, cite: "Han et al. 2024", source: "Fig. 2", note: "" },
  { id: "sorry-bench", name: "SORRY-Bench", n: 440, cite: "Xie et al. 2024", source: "Fig. 2", note: "" },
  {
    id: "xstest", name: "XSTest", n: 250, cite: "Röttger et al. 2024", source: "§5, §5.1, Fig. 2",
    note: "XSTest is an exaggerated-safety suite: its 250 safe prompts measure over-refusal rather than harm elicitation (§5).",
  },
  {
    id: "safetybench", name: "SafetyBench", n: 2100, cite: "Zhang et al. 2024", source: "§5, §5.1, Fig. 2",
    note: "SafetyBench is a multiple-choice knowledge probe; n = 2,100 is its Chinese subset, 300 questions in each of seven categories (§5).",
  },
  { id: "hex-phi", name: "HEx-PHI", n: 330, cite: "Qi et al. 2023", source: "Fig. 2", note: "" },
];

export function suiteById(id) {
  return SUITES.find((s) => s.id === id);
}
