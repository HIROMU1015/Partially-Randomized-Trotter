# 代表24候補normalization Phase 0結果（2026-11-11 JST）

承認された固定6ラウンド・24候補をSOURCE `6a75ebf967aaae2698ce6d295bf6fbec9d5e9929`で一回評価した。**全24候補のlog B・attenuation・normalization由来shot係数を記録し、数値精度未解決/overflow/underflowは0。** 9 SOURCEと7入力hashを保全した。閾値・R採択・科学的PASS/FAILは設定していない。新signal・phase/radius・十分N・C/G、候補追加・科学retryは0。公開・remote照合後mandatory STOP。

[原結果](run01/result.json)／[B/C受渡しJSON](run01/HANDOFF.json)／[全24候補CSV](run01/NORMALIZATION.csv)／[確認結果](CHECKS.json)／[読取専用検査SOURCE](VERIFY_RESULTS.py)。実行時間はexecute()区間5.005041秒（preflight除く、library importを含む）。exit0、stderr空。

## 固定条件とSOURCE

[準備REPORT](../pr_r_normalization_phase0_preparation_20261111/REPORT.md)・[PLAN](../pr_r_normalization_phase0_preparation_20261111/PLAN.json)と同一の24 ID/R/lawを維持した。実装SOURCEは6ed8c6873311085bb3b39799b268105b0bc97211、実行SOURCEは上記6a75ebf96。既存pr_r_screening、K2 finite normalizer、ABS_TIME_PER_OUTER allocation、B metadata関数を変更していない。

Linear H4/H6、1Å、STO-3G、保存tol-only DF10/DF19・weighted completion・fragment順・sector/state/H/snapshot bindingを維持。H4はS2/δ=.1/prefix1/K2、H6は4th(m5_best)/δ=1.6/prefix7/K2。h=.1/13round、保存エネルギー精度・α/β・prior・decoder・compiler/control/full measurement wrapper/P=0はmetadataのまま。今回state actionは行っていない。

保存contextからH4 lambda_R=2.8629570945298686、H6 lambda_R=0.1649333839054104をそのまま使用。これは保存decompositionのnormでR_outerとは別の量。Hamiltonianからnormを再計算したものではない。過去receipt/status/ground資格は変更せず、normalization記録をground/state/actual法則の保証へ昇格していない。

実範囲はH4 m0/6/12の12候補、H6 m4/8/12の12候補、144 piece slot。qPFはH4=1/64/4096、H6=1/16/256、qRPEはH4=1/64/4096、H6=16/256/4096。全event数=qPF R_outerを解析的に会計し、その回数のtrajectoryや行列作用を実行していない。Hamiltonian/state配列読込み・再生成、SCF/DF、固有値問題、sampling、回路build/compileは0。

## 全候補の記録

A=exp(−log B)、F=exp(2log B)。Fは誤差を無視した既存geometryのnormalization由来係数であり、**actual N・十分N・最小shot数・厳密な資源下界ではない**。表は表示丸め、全binary64値・piece配分・SOURCEはJSON/CSVを使用する。

| 系 | m | R_outer | log B | A（減衰係数） | F（normalization係数） | event数 |
|---|---:|---:|---:|---:|---:|---:|
| H4 | 0 | 2 | 0.0404893917899 | 0.960319351727 | 1.08434789034 | 2 |
| H4 | 0 | 4 | 0.0204288793383 | 0.979778376483 | 1.04170392166 | 4 |
| H4 | 0 | 8 | 0.0102378258614 | 0.989814402292 | 1.02068671598 | 8 |
| H4 | 6 | 4 | 1.30744827765 | 0.270509441439 | 13.6658026499 | 256 |
| H4 | 6 | 128 | 0.04098249418 | 0.959845932691 | 1.08541780691 | 8192 |
| H4 | 6 | 256 | 0.0204912930046 | 0.979717226831 | 1.0418339629 | 16384 |
| H4 | 6 | 512 | 0.0102456522426 | 0.989806655657 | 1.02070269267 | 32768 |
| H4 | 12 | 4 | 83.6766897695 | 4.56746648398e-37 | 4.79346169932e+72 | 16384 |
| H4 | 12 | 8 | 41.9341347284 | 6.14096587517e-19 | 2.65171369697e+36 | 32768 |
| H4 | 12 | 8192 | 0.0409826189029 | 0.959845812976 | 1.08541807766 | 33554432 |
| H4 | 12 | 16384 | 0.0204913020103 | 0.979717218008 | 1.04183398167 | 67108864 |
| H4 | 12 | 32768 | 0.0102456808086 | 0.989806627382 | 1.02070275098 | 134217728 |
| H6 | 4 | 11 | 0.016762898189 | 0.98337681742 | 1.03409411927 | 11 |
| H6 | 4 | 16 | 0.0113844324945 | 0.988680124942 | 1.02303005416 | 16 |
| H6 | 4 | 32 | 0.00534535907357 | 0.994668901937 | 1.01074806806 | 32 |
| H6 | 8 | 16 | 0.182150919912 | 0.833475542868 | 1.43950864963 | 256 |
| H6 | 8 | 128 | 0.0210616502146 | 0.979158597371 | 1.04302307602 | 2048 |
| H6 | 8 | 256 | 0.0105187856392 | 0.989536343321 | 1.02126042098 | 4096 |
| H6 | 8 | 512 | 0.00525748040085 | 0.994756315961 | 1.01057043728 | 8192 |
| H6 | 12 | 16 | 2.91441471859 | 0.0542357649303 | 339.960500949 | 4096 |
| H6 | 12 | 32 | 1.36841192283 | 0.25451082161 | 15.4378740749 | 8192 |
| H6 | 12 | 2048 | 0.0210279463714 | 0.979191599335 | 1.04295277061 | 524288 |
| H6 | 12 | 4096 | 0.0105139450456 | 0.989541133275 | 1.02125053401 | 1048576 |
| H6 | 12 | 8192 | 0.00525696721548 | 0.994756826455 | 1.01056940006 | 2097152 |

記録範囲ではR増加に伴う減衰抑制を確認した。H4 m12のR4→8でもAは約4.57e−37→6.14e−19で、R8192では約.960になる。一方event数は16384→32768→33554432。これはcompiled Cの比例則ではなく、R増加に伴う回路構成上の負担をC側で検証するためのmetadataである。

qPFに比例させたR bandでは、H4 m6とm12の対応するlog Bが約.04098/.02049/.01025、H6 m8とm12では約.021/.0105/.00526となった。これは小τのR∝qPFという目安と整合する有限記録で、厳密な時間不変性・全Rの単調性・探索範囲の十分性や最適Rを示すものではない。

## H6配分と丸め診断

H6は既存11 signed pieceの値/順序・負時間を保全し、全Rを一outerへ配分した。Σr_j=R、r_j≥1、event数=qPF Rを全行照合した。R16は`[1,2,1,2,1,2,1,2,1,2,1]`、R32は`[3,3,3,3,2,5,2,3,2,3,3]`。整数剰余のtieをpiece index順で扱う既存規則により、配分が左右対称とは限らない。新しく対称化せず、そのままC/N側へ渡す。全配分はCSV/JSONに保存した。

normalizerの1 ULP累積log感度maxはH4 m12/R32768の5.960464477192118e−8。これは丸めに対する感度診断で、rigorous error boundやconfidence intervalではない。小τ主項は診断としてのみ保存し、評価値の置換・採否には使用していない。全表示はFINITE_BINARY64、NUMERICAL_NORMALIZATION_UNRESOLVEDは0。正のAや有限なFを位相・radius・N成立と混同しない。

## 検証・保全・HANDOFF

CHECKS.jsonで固定順序の24 ID、law/全identity、allocation/event数、logと指数表示の整合、JSON/CSV/HANDOFF一致、SOURCE9/入力7/旧H4-H6 N原本を確認した。新しいnormalizerでの再実行やprecision retryは行っていない。準備時の人工/mock10件と元screen12件はSOURCE不変の保存記録を再利用した。これは主担当自身の検査で、別AIによる独立科学レビューではない。

HANDOFF.normalization_evidence_valid=trueは、Phase 0の完全な数値記録とSOURCE/input保全を示すだけ。全rowでN/C/G=null、phase_radius_N_status=NOT_EVALUATED、recommended_R/scientific_candidate_PASS_FAIL=null、full_RPE/R採択は未資格。費用evaluatorやR optimizer、providerによる候補追加を呼んでいない。保存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/過去STOP、B/C作業を保全した。

B/Cは[準備HANDOFF](../pr_r_normalization_phase0_preparation_20261111/HANDOFF.md)の形式で同じrequest.identity/law_binding・R/qPF/δ/prefix/K/signed配分とcompiler条件を照合する。normalizationだけから旧Rのbias/N/Cを新Rへ付け替えず、AやFをCへ二重乗算しない。未評価N/Cや未検証domain/actual法則は不足のまま残す。

次はユーザーが詳細N/Cの候補・実行範囲を判断し、B/Cがこの記録と価格domain/必要な検証範囲を照合する。Phase 0の完了をPhase 1・新N/C/G・R最適化の実行許可に読み替えない。今回の公開・remote SHA/主要成果物照合後mandatory STOP。
