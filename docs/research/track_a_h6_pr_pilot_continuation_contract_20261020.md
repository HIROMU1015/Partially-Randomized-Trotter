# H6 PR DESIGN_PILOT：74本の独立受入と元の残り322登録の継続契約

2026-10-20、利用者の「再計算を開始してよい」を、直前に提示した保存74本の受入と固定残り322本/最大644軸の取得に対する明示承認として記録する。旧run01のSTOP・exit1・172 compile予約は変更・返却しない。本契約内の自動retry・top-upは行わない。

対象はH6直鎖1 Å/STO-3G、tol-only 1e-8/DF19、保存Hamiltonian/state/reference、prefix5、T=.8、P=0、S2/Yoshida4/m5_best、q1/2/4/8、K4、ABS_TIME_PER_OUTER=32。12群×33本の元登録396本を維持する。旧registry・元SOURCEによるseed導出・namespaceを再検査し、新SOURCEでseedを引き直さない。pilot以外の標本、探索、独立確認とのpoolingはしない。

旧SOURCE `de5485d4b7c6de25dd9b6137c9a12bce697242c6`、旧plan `4b9dd672040b12d82efb5093c413894a32e862bfac99904524fb616ff8b23676` の保存74本を、新しい受入receiptにより採用する。native8 gate・4404片の生成law/seed・444 archiveの原SHA/完全復元・result集合SHAを起動前と終了時に確認する。旧396本全体をPASSに書き換えることはしない。

元396登録から固定74本を除いた322本だけを新出力へ取得する。partial cosineと新sineを結合せず、未完了登録は同じ元seedでtrajectory両軸を取得する。予定は322 trajectory、3021 request/片合成、322 native evolution/low/high/oracle、644 wrapper/compiled作用・compile軸、32288 RTE event。新reference/SCF/DF/state生成、signal、shot、統計、費用会計、fit、ランキングは0。希少eventの強制・棄却・結果依存の補充はしない。

cutoffは修正SOURCE `e7595e8ff543a0d0f5e1ea91dec14283b61f9c32` と同じ、factoryのsymbolic lambdaおよび演算順序によるτを使用する。係数・signed schedule・確率法則・K4は維持する。compilerはQiskit1.3.0/rz-sx-x-cx/opt1/seed17、symmetric-directional full Hadamard wrapper、cosine受入後sine、scalar-onceを維持する。compiled状態差CT01は1e-7未満、その他native/oracle/Hadamard等は1e-9未満とし、不一致なら当該継続契約をSTOPする。

軽量高速化はH6専用opt-in。親で共通contextを1回作りforkで共有し、SHA検査付きJSON cacheを使う。fork前の単一thread検査、context構築後のfresh gateを維持する。並列数は実空きCPU/RAMと同入力の2倍RSS proxyから決め、過去28workerを恒常的上限にしない。4倍圧縮容量＋16MiB/trajectory、一時raw/partial/復元、inodeのfresh admissionを行う。容量proxyは上界ではない。ホストPSIや他worktree資料更新は観測のみとし、共有設定・他jobは変更しない。

新QPY/NPZのzlib/DEFLATE level1保存は、原SHA・archive SHA・復元PASS・manifest永続化後だけ新一時rawを解放する。旧rawは変更しない。native検証とbyte復元検証を区別する。新SOURCE・独立grant/ledger/marker/STOP・外部wait4 exit/reap・worker回収・全call台帳・終了byte監査を保持する。

終了監査は新322本と保存74本をそれぞれ検査して、元396登録の被覆を参照表で結合するだけであり、平均・分散・MCSE・SE・n_E・shot・費用順位は計算しない。AD03・正式shot政策は保留。正常/異常終了を問わずmandatory STOP。次の担当はCodexの技術的終了報告、続いてユーザーによるpilot数値評価の別承認判断である。
