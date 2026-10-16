# Track A V2：H6探索PFの最終限定照合と現行残差設計のNO_GO候補

**比較可能と認定できるH6探索PFは0件だった。既存の固定条件でfit可能なstratumは0件のままであり、現行V2残差設計を`NO_GO_CANDIDATE_CURRENT_RESIDUAL_DESIGN`として整理する。利用者の指示に従って広域回収を終了し、fit・残差計算・新規compileは起動しない。**

branchは`track-a-v2-residual-analysis-readiness-20261016`、起点は`373d544edb675bd873c4236c5d1c656695507cfb`。[前回の全体照合](track_a_v2_provenance_reconciliation_20261016.md)から対象をH6探索PFだけへ限定した。[判定表・SOURCE receipt・監査手順](../../artifacts/resource_applicability/track_a_v2_h6_pf_final_reconciliation/2026-10-16/README.md)。

判定対象は、現契約によるuniform A/B別・軸別の保存RZ単価共通fitの実行可否である。832理論行と保存compiled結果は既存成果として保全する。別の解析設計の採否は利用者判断に残す。

## 限定した対象と確認方法

linear H6、1.00 Å、STO-3G、保存DF rank19、12 spin modes、T=.8。対象は旧B0 prefix5/10/15のS2とfull prefix19のS2/S4、q1/2/4/8、計20 PF候補である。B0 prefix5の4候補は未測定、残る16候補は保存両軸を持つ。V1追加格子・p7実測・B0I・H6確認・H8の原rawを追加走査していない。

元のcandidate/task/seed、manifest digest、worker SOURCE、terminal、trajectoryのRTE非該当、両軸wrapperのcontrol/compiler/fingerprintを照合した。SOURCEの実行経路、profile検査、preparation生成と保存範囲をcodeとして読んだ。native builder、preparation decoder、数値library、compilerはimport/実行していない。保存費用・signal・確率を計算に使わず、既存理論費用JSONとcompiled接続gzip、NPZは開いていない。

## 最後に回収できたことと、不足が残る理由

|対象|回収・照合結果|結合済みPRとの比較可能性|
|---|---|---|
|初期SOURCE `54986a9f…`の15 PF候補|task・両軸とpackage/version・compiler optionsは一致。manifest、worker、wrapper、terminalには固定compiler profile receiptへの結合がない|profileとbasis lineageが未解決。後続profileを遡及適用しない|
|SOURCE `6a924e52…`のfull S4/q8 1候補|旧profile SHA `4e32bb9b…`に結合。保存profileファイル・公開Git blobが一致し、原workerのprofile検査経路も確認|profileは結合できたが、各wrapperのprepared basis全記述との独立結合は未完了。fitへ昇格させない|
|共通native構成|両PF SOURCEとPR provider `87b6b9f4…`の共通library/native v5/prepare実装8ファイルがbyte一致。PFはRTEを含まず、blockごとに閉じたnative構成と明示scalar phaseを使用|SOURCE意味論の対応。全runtime basis・binary/profile同値の代替証拠ではない|
|未測定B0 prefix5の4候補|元の未測定statusを保持|新compileや0費用で補わない|

PF cost workerは各taskで`_prepare_discard`又は`_prepare`を呼び直す。保存trajectoryにはPFのRTEがないことは記録されるが、そのcost taskのpreparation hash・全block/basis記述は保存されていない。signal側のprepared summaryは参照できるものの、cost workerの準備実体そのもののreceiptではない。

signal側のprefix10/15/full19の保存preparation hashは、PRに結合したroundoff archiveの対応root hashと異なる。hash差だけで物理的差を断定せず、同一性も推定しない。archiveの既存export receiptは`deterministic_preparation_unchanged=true`、政策は`unchanged server native`と宣言している。一方で`old_preparation_hash_exact=false`も保存されており、旧初期PF全候補の独立したruntime結合を与えるreceiptではない。

したがってprofileが確認できたfull S4/q8にも、この宣言を超えた全basis同一性の認定は付けない。既存wrapper fingerprintは原receiptとのidentity確認に使い、理論側に欠けるprepared basisの記述を復元したことにはしない。新しいbasis/行列/Gaussianの生成や数値照合は行っていない。

## 固定条件への影響

事前登録のhash分割を変更せず、今回のPF metadataだけを分類した。

|確認範囲|train PF|holdout PF|意味|
|---|---:|---:|---|
|全測定PF（来歴を無視した前回の上限）|13|3|適格性を認定しない上限|
|旧profileへの結合を確認できるPF|1|0|full S4/q8のみ。basisの昇格ではない|
|全critical来歴が結合済みの比較可能PF|0|0|今回の実際の適格件数|

holdout PF3候補は`B1_p19_q2_4th`、`B0_p15_q8_2nd`、`B1_p19_q8_2nd`で、すべて初期SOURCEのprofile未解決群に属する。profile確認済みの`B1_p19_q8_4th`はtrain側である。

既に結合したH6探索PR17候補のtrain11/holdout6は変更していない。比較可能PFが0件のため、新しい共通fit stratumの登録・予約・数値rank検査は行わない。さらに、profile確認済みの1候補についてbasisの宣言を仮に認めても、PF train1/holdout0という必要条件の上限にとどまり、双方各2以上を満たさない。これは1候補を適格へ昇格させた判定ではない。

探索/確認の統合、B0IやV1実測の追加、別profileの混合、別hash・分割探索、PR/PF最小件数の緩和は行っていない。来歴回収を続けてfit対象を作る作業は、この限定調査を最後に停止する。前回確認したH6確認・H8のholdout PR=0という制約も変更しない。

## 台帳・保全・公開後STOP

更新private台帳はSHA-256 `8f7f07d00446100f7100f27c4c94808781826ba79a442daf756fc62457a5aaa4`で固定した。変更したのはH6探索PF20 entriesの最終証拠記録と、full S4/q8のprofile結合だけである。PR・確認・H8・B0Iのentryは不変で、全体の条件付きmetadata結合済み25件・未解決237件は維持した。旧台帳・runtime config・grantへは設置していない。

141入力receiptのbytes/SHAを記録し、原SOURCE/固定profile・両軸・候補・manifestのassertionと、対象を20 PF entriesへ限定するguardを確認した。primary agent自身の読取専用local checksであり、独立レビューではない。旧43人工testsや科学campaignの反復はしていない。

旧667保護ファイル、残差実行SOURCE9ファイル、V1・旧V2・理論worktreeのHEAD/status、旧V2未追跡scriptを前後照合して保全した。科学runtime・affinity・他jobは変更していない。公開対象は本報告、20候補のmetadata判定、SOURCE/hash receipt、監査手順と索引だけで、private台帳・config・分子配列・保存費用値は含めない。

**現契約の残差解析はNO_GO候補とし、別設計の採否を利用者が判断できる状態で終了する。新fit・残差・compile・signal・sampling・law/primitive評価、新config/grantの発行は0。commit・push後にremote SHA/treeを確認してmandatory STOPし、広域回収・次の解析へ自動移行しない。**
