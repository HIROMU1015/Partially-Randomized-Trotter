# 2026-11-13：R3 decoderとB S3接続

独立R3 decoderの4候補branch選択を実装し、元scalar/vector各tie規則とBのcount/RNG/loss/CIを保全。新12＋旧20の32技術test PASS。人工M12/512trialで旧vector比96.3倍、B batch接続664.6倍、実H4/S3/保存再解析/C/G0。旧vector exact-tieのみ互換fallback、公開照合後STOP。

監査a8f32abb…、B170047ef…を固定し、最新Bbf515a976…も対象統計blobが同一と照合。元A bba150834…の物理input・科学資格を保全し、Bのbranch/worktreeを変更せず任意batch hook差分を提示。NumPy argsortのexact-tie順がscalar argminと異なるため、二つの互換APIを分離した。

[REPORT](../rpe_decoder_r3_20261113/REPORT.md)／[HANDOFF](../rpe_decoder_r3_20261113/HANDOFF.md)／[人工benchmark](../rpe_decoder_r3_20261113/BENCHMARK.json)。実データのdecode・統計評価は行っていない。

次：Bが最小hookを取り込み、ユーザーが固定SOURCE・target/loss・統計精度と有限S3実行を別途承認する。Aは公開照合後mandatory STOP。
