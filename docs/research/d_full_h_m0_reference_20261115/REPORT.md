# Full-H κ2 m0：1 normal path取得、参照単価はnullでSTOP

**予定どおり1 pathを一度samplingし、同じpathのX/Yを各1回compileした。元観測RZは各1,062。**
準備照合とnativeの7数値gateはPASS。ただし既存P1総合gateの `parity.replaced_controlled_zz > 0` を満たせず、
両参照単価 **C_0X=C_0Y=null**、受入保留として終了した。再抽選・追加path・retry・gate変更は0。
観測compile値を採択済み単価へ昇格せず、証拠と理由を保存・公開してmandatory STOP。

## 固定入力・実行

利用者が採択した規則はFULL_H_KAPPA2_M0_NORMAL_PATH_REFERENCE_V1。
C提案4a1cb0d7364fece475437749129fdce740fd8fc4のPROPOSAL.json/HANDOFF.mdと、
61f249e6e18d5afab9ba8797bef5e8f9f14e29deのADMISSION.json rows[0]/rows[1]へ結合した。
科学SOURCE/inputは `7f85f20e38af4df1a649ac239e57bfe76d0899ee` にcommit/push・remote照合後、別checkoutで固定。
結果SHAは `b25d7da6dedebc4cd22e8d6f00e6de75d7072292`。

H4直線1.00Å/STO-3G/保存tol-only DF10、one-body込みFull-H pure RTE/L_D0、finite K2・κ2、P0。
physical ID FULL_H_2a9a68a11d08af234a8c、m0、T=.1、q1、R9。
cohort D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1、index0、seed15612141344395026302。
独立path数1、X/Yは同じpathの別測定軸である。

§10.5はCの提案REPORTの既存照合を継承し、提案公開後の関連ref・既知private rootのfile名・関連jobを確認。
同条件の既取得0、同一進行job0。旧D7、P1、K4/T.8、T2.0、PauliやDF12-LD3/seed7のm0は転用しない。
[PREFLIGHT.json](PREFLIGHT.json)。未知archive/他hostを網羅したとは主張しない。

sampling前にD7 preparation hash92325333cfe8c5e6c5f9bf69cdfc0ea6538f8eaa66327a3820d390dc2e8aa100、
H/state/snapshot、DF fragment順0..9と元prepare_full_h SOURCE、scalar定数/抽出/合計を照合PASS。
component順はRESULTのchild.context_checkへ保存した。scalar合計−0.33147781341681615を元総T一回の経路で適用。
Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、symmetric-directional、新ZZ lowering、測定付きfull controlled Hadamard wrapper、
state preparation費除外を保持した。保存reference contextのT=.8表示は保存referenceの条件で、取得T=.1とは区別する。

## 実測・受入状態

| 軸 | 元compiled RZ（観測証拠のみ） | 参照単価 C_0a | 受入 |
|---|---:|---|---|
| X | 1062 | null | HELD：必須ZZ実績条件不成立 |
| Y | 1062 | null | HELD：必須ZZ実績条件不成立 |

96 shot・RTE normalizerの乗算0。全13roundの新比較シナリオ・総費用算出0。
各観測値はそれぞれの元compiler recordから読み、X/Yをコピーしていない。
X fingerprintはb23187eaf78d4fd7a3614460a1c03f22ea83311425478375d4a71a21a25fc332、
Yは5c853c3469cb65c688bce01e3a9bb9c4ca8d6b6666ae3d0929944f58aca0b16c。
元保存wrapperの測定末尾を読むことで、Xはh/measure、Yはsdg/h/measure、ancilla8/9qubit/1cbitへ個別帰属した。
この読取ではsimulation・build・compile・RZ再計数は行っていない。元QPY/compile JSON bytesとhashを保存。

予定1 path/最大2 wrapper、実行1 path/2 compile、事前価格再利用0、compile receipt欠測0。
normal_sampler_calls=1、元cache_hits=0。受入単価0/保留2、7数値gateのFAIL0、総合gate不成立1。
科学retry・seed差替え・追加path・閾値緩和0。

## 停止理由と確認できないこと

保存eventはorder列 `[0,0,0,0,0,0,0,0,0]`。各事象にはcentral rotationが存在し、そのsupportはすべて1 qubit。
one-body適用は6件、two-qubit ZZ適用は0件である。order0を空の回路やscalarだけの回路として扱わない。
元nativeのlow-control/high-control leakage/independent occupation vector、Hadamard X/Y、compiled vector X/Yの
7数値gateはすべて元閾値未満でPASS、phase_alignment=false。

既存 `scripts/run_full_h_kappa2_p1.py:pilot` は、数値検査に加えてparityのZZ置換実績が1件以上であることを
総合okの条件とする。この保存pathには該当操作がなく、条件が不成立になることを保存event metadataと固定SOURCEから特定した。
数値gateの不一致と、実行coverage条件の不成立を区別する。[STOP_DETAIL.json](STOP_DETAIL.json)。
理由特定のための回路再構築・gate再評価は行っていない。

acquire_pathは総合gate不成立で例外を投げ、個別のparity/scalar-event/XY戻り値を返却前に失った。
その個別値はNOT_PERSISTEDでありPASSと補完しない。元native result/event、context、両compile/QPY、traceback、
progress、終了codeを保全した。固定条件を緩和して再度取得することは行わない。

## SOURCE・証拠・資源

既存normal sampler、P.acquire_path、native/wrapper/compiler/gateとA.accelerated_pathを再利用。
X/Y別process、既存exact gate cache/fast fingerprint、親observerを使用し、非同期stack dumpは追加しない。
元D7/F1/Cの原本・旧STOP・共有環境には変更を加えていない。

追加人工4件PASS、import-only23 module PASS。実H4試運転0。
科学job wall7.84秒、user25.72秒/system12.66秒、GNU time maxRSS368136KiB、親/子exit2・signalなし。
元native trialは1.277秒、peakRSS375005184 bytes。maxRSSを同時worker合計memoryとは扱わない。

公開結果SHAからraw結果/native/event/context、元2 compiler JSON、元4 QPYのhashと個別軸対応を照合した。
[PUBLICATION_CHECK.json](PUBLICATION_CHECK.json)はD1の保全・来歴照合であり、参照価格の受入やC科学的受入ではない。
軸・null単価・観測値・seed・SOURCE・各receiptの固定SHA/path/selectorは[HANDOFF_C.json](HANDOFF_C.json)。

## ロードマップ差分・終了

「m0経験的参照規則は採択済み、指定normal pathの2軸compile証拠は保存したが、必須ZZ実績条件不成立で単価2軸はnull保留」。
今後必要なのはCによるこのcoverage条件の扱いの判断で、今回D1が条件を変更・価格を再資格化することはしない。
元F1のm0/全額nullとD7の資格を保持する。新しいm1〜m12結合シナリオ・96倍・fit/Q3・E[C]/CI/N/G/winnerは未実施。
ロードマップ本体は統合担当が更新する。結果・HANDOFFのcommit/push、remote一致・公開証拠照合後mandatory STOP。
