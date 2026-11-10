# R可変C価格のB受入interface（2026-11-11 JST）

[REPORT](../rpe_r_price_intake_20261111/REPORT.md)／[HANDOFF](../rpe_r_price_intake_20261111/HANDOFF.md)。B `d5d9823e`の既存moduleを変更せずに、Cの`describe_request`（byte一致）と固定C1 PLAN（`c596e88d`、Cの`validate_plan`）を接続した。C1の8セルはB `build_menu/request_context`のfull identityと一致した。lawは、N側（Aのideal mean）とC sampler（PCG64/CDF）を、`rte.py`の同一blobを根拠に対応付けて両方を保持した。samplerとideal lawの距離は未資格。

公開後のCALIBRATION_EVALUATION/VALIDATIONの価格recordは、copyと検査だけで受け入れる。検査はstage順、Git凍結、校正3セル/system、検証の分離、cohort・n_paths・source・model domain、null不確かさ。保存校正、新R校正、独立検証、model予測、C1外の欠測を`price_kind`／`R_price_status`で区別し、rareは別unitで扱う。固定R4/R16の価格を新Rへ移すとproviderが停止する。

130技術test（129 PASS＋任意skip1）、新科学計算0。C公開の`2e6a0cdf5`（Phase 1完了）は指示の前提に従い取り込んでいない。次は指示を受けてからpinned commitで取込みを行う。公開照合後mandatory STOP。
