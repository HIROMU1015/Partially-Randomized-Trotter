# H6 PR DESIGN_PILOT：保存74本の独立受入・残り322本を一回起動

利用者の再計算承認に基づき、専用継続契約を実装・検査し、H6の元固定登録322本/最大644軸を起動した。保存74本は別受入receiptにより保持する。元396登録・12群×33本・元seed/namespaceを変更しない。旧run01のSTOP・exit1・172予約は保持する。

科学SOURCE：`0ff0f70491423d2ef7cb99b006e6b054a8b17b3b`。plan SHA：`2eae0538667a648a139f75edb8b9787e2141c6b08554617ba211ea08993166db`。[実行契約](track_a_h6_pr_pilot_continuation_contract_20261020.md)、[起動snapshot](../../artifacts/resource_applicability/track_a_h6_pr_pilot_continuation_run01/2026-10-20/start_snapshot.json)、[検査記録](../../artifacts/resource_applicability/track_a_h6_pr_pilot_continuation_run01/2026-10-20/technical_checks.json)。

H6直鎖1 Å、STO-3G、tol-only1e-8/DF19、保存Hamiltonian/state/reference、prefix5、T=.8、P=0。S2/Yoshida4/m5_best、q1/2/4/8、K4、outer32、signed schedule、symmetric-directional full Hadamard wrapper、Qiskit1.3.0/rz-sx-x-cx/opt1/seed17を維持する。CT01は1e-7未満、その他native/oracle/Hadamard等は1e-9未満。

新契約の人工8件、修正・高速化の回帰16件、実configの改変拒否11項目がPASS。起動前に保存74本/444 archiveの原SHA復元と4404片のlaw/seedを再照合した。新SOURCE/input/compiler/approval/registryのfresh gateがPASSし、新grantを一回消費した。contextを親で1回準備し、fork前の単一thread確認と構築後の資源gateを通過した。worker数は30。SHA付きJSON cache・長い固定schedule優先dispatchも有効である。全体高速化率はまだ測定していない。

保存容量proxyは約216.9 GB。4倍圧縮＋16MiB/trajectoryと、最大affinity64 slot分の一時raw/partial・復元領域を予約する。64 workerを強制する意味ではなく、actual fresh CPU/RAMに基づく並列数の増加も収容するためである。proxyは実PR/rare出力の上界ではない。旧rawを削除・変更せず、新規rawだけをlevel1 archive・manifest永続化・復元確認後に解放する。

準備attempt01はSOURCE引数の誤りを静的gateが拒否した（科学0、grantなし）。attempt02はscratch予約調整前の未消費grantで、科学0・起動なしのまま保持した。修正後のSOURCEに結合したattempt03だけを一回科学dispatchした。これは科学workerの自動retryではない。旧保存結果・STOP・失敗準備記録は消さない。

この文書は起動記録であり、取得完了・全native PASS・最終exit/reapを認定しない。未完了部分のcosine/sineは同じ元seedで両軸取得し、旧partial出力を結合しない。標本追加・希少event強制/棄却・本探索や独立確認とのpoolingは行わない。統計・MCSE・n_E・shot・会計・順位・AD03は保留。

専用supervisorはチャット終了後も継続し、native exit/reap・新322本の終了監査・保存74本の再監査・元396登録の参照結合を保存してmandatory STOPする。異常時もSTOPして自動retryしない。次の担当：Codexが次回確認時に進捗・終了結果を報告、ユーザーが取得終了後のpilot数値評価を別途承認するか判断する。
