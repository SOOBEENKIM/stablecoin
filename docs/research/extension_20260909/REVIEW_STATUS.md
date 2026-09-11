> GitHub 탐색용 사본입니다. [체크섬이 보존된 원본](../../../research/extension_20260909/REVIEW_STATUS.md)의 내용과 과거 시점 표기를 유지하고 문서 링크만 변환했습니다. 실행은 [현재 재현 안내](../../REPRODUCING.md)를 따릅니다.

# Review status and analysis history

The main extension's numerical forecast specifications were fixed in `PROTOCOL_KO.md` before their evaluation. `RUN_MANIFEST.json` hashes the four inputs, the protocol, the engine and its preprocessing dependency. No main forecast configuration was added or tuned after viewing extension performance.

The complete overall dataset and the first pilot's January-March performance had already been inspected. Consequently this extension is not an independent holdout experiment, even though each forecast obeys chronological information restrictions. The final draft explicitly labels the study exploratory.

The comparison of reference definitions, one-coin deletions and price-grid history was included in the planned measurement diagnosis. The detailed February one-tick sensitivity, the interpretation emphasizing reference sensitivity, and the illustrative episode choices were developed after inspecting descriptive diagnostics and are exploratory. They are not causal policy estimates or independently confirmed hypotheses. The first pilot and all extension comparisons are retained, including negative results.

The prespecified strong-ML claim criterion failed: the primary calibrated local tree did not reduce loss by 3% against the calibrated linear model, and its monthly comparison was unfavorable in all three test months. Near-nominal coverage after calibration does not rescue that algorithmic claim. The draft therefore centers reference sensitivity, forecast maintenance and limits of the tested information sets.

Completed: all five planned cells; information ablations; raw/calibrated and updated/frozen comparisons; 14/28/56-day window sensitivity; alternative block lengths; nonoverlapping six-hour origin phases; favorable-day influence checks; three independent timing tests; raw-data/protocol/engine hash verification; paper figures and supporting CSVs; eight-page English review PDF and editable Markdown; Korean findings and scope assessment.

Not established: independent out-of-period or out-of-venue generalization; a causal tick reform effect; stablecoin-specific latent demand; executable profitability; a new ML algorithm. The precise article novelty is a bounded literature-informed judgment, not proof of worldwide firstness. Several recent papers were accessible only through publisher abstracts and public sections.

The PDF is an author-review draft. Final venue-template compliance, author review, the relation to the domestic manuscript under review, and any applicable submission-policy confirmation remain before external submission. The domestic manuscript's arithmetic/time-axis corrections are a separate outstanding manuscript task; replacing its original outcome with this new target is not a correction of the same estimand. No external submission or message has been sent.
