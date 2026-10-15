# 2026-10-16 Native DFモデル：実Qiskit接続の技術検証

利用者の指示に従い、固定1ea42fbf…から独立worktreeを作成。
[主要資料](../track_a_native_df_qiskit_connection_validation_20261016.md)へ42人工testsの接続結果を保存した。
既存`.venv311`にQiskit 1.3.0があることを確認し、install・環境ファイル変更なしで使用した。

旧カウンタは手入力されたzero RZ項を除去していたが、実appenderは保持していた。
修正前の差を保存し、新counterだけを限定修正。通常producerは上流でzero項を省く。
元native/common source、旧結果、旧receipt、科学的比較条件は変更しない。

Gaussianのliteral gate列と物理Fock liftのvacuum phaseの違い、
native PF builderがemitしないモデルB相殺、別occurrenceとflattened列のreuse差を分類した。
これらの定義差を数え上げバグとして無理に吸収しない。
一般identity phase期待値のboundとcompiled母平均未認定を維持する。

主担当の技術チェックであり、GPT詳細科学レビューは開始していない。
実H6/H8への適用・保存費用再解析・新科学計算・server操作・新grantは0。
比較契約への採用範囲は後段のGPT判断事項。公開後MANDATORY_STOP、next_stage_authorized=false。
