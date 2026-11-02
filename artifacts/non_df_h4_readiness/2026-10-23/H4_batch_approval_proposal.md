# H4 pilotの将来の一括承認文案（現在は未承認）

以下は利用者が将来採用・承認する対象を具体化した文案であり、実行権限を与える記録ではない。
`H4_execution_contract_DRAFT.json` のstatusはDRAFT、grant_valid=false、dispatch_enabled=falseのまま。

「linear H4・隣接1.00 Å・STO-3G・charge0/spin0・active/frozenなしの固定4方式pilotについて、
別途完成して提示された実装SOURCE・本契約のdigest・環境を固定した後、RHF1回、同じMO積分からPauli-JWとtol-only1e-8 DFを各構築し、
nonDF Nalpha=Nbeta=2 ground1回・DF sector照合1回・FCI照合1回、固定prefix規則によるPF/PR計4caseを各1回実行することを承認する。
T=.8/q8/δ=.1/S2、PR r4/K4、各PR256 paired trajectories、固定5精度、既存Hoeffding/経験的allowance、compiler opt1/seed17を使用する。
入力生成後の実rank・項数・prefixは事前固定規則から一意に決め、case release前に入力receipt/hashへsealする。
生成・照合・signal・全wrapper compile・C/N/G・数値検証・保存・ローカルcommitまでを一括で進め、工程ごとの再確認は不要とする。
one-shot、新grant・固定claim directory、失敗peer停止・retry/resume禁止、精度不適格N/Gはnull、既存科学成果保全、完了後mandatory STOPとする。
H6以降・候補探索・標本増加・固定契約変更・追加独立確認・別の希少event科学run・push・詳細科学レビューは含まない。」

実行前に必要なのは、H4専用molecular runner/dimension gate、production drawのevent監査hookと人工同値性検査、
そのclean SOURCE/環境seal、本契約の正式採用、および上記を対象にした利用者の明示一括承認である。
この文案を保存しただけでは、新しい科学的な法則・推定器・希少event保証は採択されない。
