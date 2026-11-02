# Angle・cache・setup・output・seed候補

固定K_root=256、H=160、η=ρ=10^-12、ε_axis=1/200、primitive_error=10^-6、
α_axis=1/320、元16-axis familywise≤1/20、root/ceil/phase/orderは変更しない。
保存Nとbiasの認証を再計算しない。

## τ、Kφ、Jは未採択

F3 `make_request` は正のτ、正整数Kφ/Jを必須とし、`enclose_atan` は
sqrt(1+r²)の外向き区間→半角z→両端J項のalternating atan remainder→代表値radius≤τを検査する。
固定精度一回のみ、不足時STOP。値の自動探索／精度refinementを追加しない。
G7の固定K/Hだけからτ/Kφ/Jは一意に定まらない。準備版に便宜的な数値defaultはない。
mock中のτ=1/16,Kφ=8,J=4は人工fixture専用であり、候補science値ではない。

今回driverが受理する第一候補は**数学的enclosureだけを取得し、物理RZへ代表値を渡さない**方式。
Layer Aは既存ratio取得のみ、Layer Bはacceptedごとに既存APIのenclosureを取得する。
元の保存RZは理想atan(ratio)に対する過去のstrict certificateとして引用するだけで、新native取得／再評価はしない。
この範囲では新しい物理誤差をG7 biasへ加えていない。一方、計算費用を定めるτ/Kφ/Jの具体値の採択は残る。
大きいKφはroot/有理operandを、大きいJは級数とevidence量を増やす。同じ許容条件でも実費が異なり、
SOURCEのper-event radius check合格を実G7で確認するのは別途承認された初回correctnessからである。

別の合法なscopeとして、representativeをRZへ渡す場合は、Rz(θ)=exp(−iθZ/2)に対し
角度radiusのoperator差≤τ/2を既存strict合成誤差へ三角不等式で足した値が10^-6以下になる証拠等を要する。
二つのRZからhelper controlled rotationへの既存factor2も維持する。保存epsilonが上限まで使われているなら
正のτを無条件に追加できない。保存marginの数値再解析や新allocationは今回未承認。
このscopeは現在driverが拒否し、別の採択／固定SOURCE／実行承認を要する。
F3 enclosureを物理RZ保証と呼ばない。C1 provider選択は含めない。

## Cacheとsetup

第一候補はlaw/packet/angleの新runtime memoを無効化し、既存保持stateだけX/Yで再利用する。
Full.sampleのlocal packet一回とF3角度再queryを残す。canonicalのconditional law再構築も残す。
F3 API全体をcache hitで飛ばすことはしない。従って架空のwarm hitは0件、lookup自体も実行しない。
別helper `ChargedCache` は人工hit/miss/lookup/acquire/insertの課金を検証済みだが、本候補のproductionで使ったと偽装しない。
将来cacheを採択するなら全方式へ同じ機会を与え、実key/precision/context一致、cold取得、evictionとguard保持を別固定する。
OS file cacheは操作せずuncontrolledとして記録する。root共有cacheや他job cacheを消さない。

setupは共有budget file一回load、24 arm/layer/phase-pass constructor、うちFull F3 prepare3回、canonical preflight9回。
Full F3 prepareはconstructorを内包するため追加constructorとして二重課金しない。
X/Yのconstructor再実行は0、identity照合は各pair4回。各方式は同じ共有条件を持つ。
source checkは実読込/hashとして残し、Full内部F3と共通overlayの双方の実費を計上する。

## 保持・保存

全trialで元event/audit/angle証拠を実際に生成・hashし、最初の16/axisとcorrectness全16/axisは完全文書を保持する。
それ以後はtrial id/kind、event/evidence/angleのSHAとgenerator CPU/wallをcompact rowで保持する。
元のphase・word順序・weightを含む正確event traceを逐次hashし、A/B・両passとcorrectness prefixを照合する。
全Nの完全event表は保存しない。hashだけから全eventを復元できるとは言わない。保持方式の採択は必要。
RAMはN件のevidenceを保持しない。fine/coarseもaggregateだけを保持する。

per-pair JSONLはexclusive new file、append/resumeなし。exact int/Fractionはhexタグで桁数制限を変更しない。
chain SHAと実file SHA、行数/bytes、partial failure prefixを残す。
serialization、chain hash、write、file hash、flush、finalizationをそれぞれ課金する。
4096 rowsごととclose時flush、Python無buffer write。fsync/power-loss durabilityの保証はしない。
result stream全体64GiB、log候補256MiB（通常loggerはなく、小さいfailure/receipt metadataのみ）。
prefix肥大／I/O失敗ならSTOP。上限は予算でありdisk予約ではない。自己receipt最終書込費はnull、費用不明を0にしない。
OS kill時は完全な行までをprefixとし、成功／全N completionと扱わない。

## 新しいbitstream候補

`G7.DeterministicBits` の固定SHA256-counter規則をそのまま使用し、
namespace候補`B3_FULL_N_NEW_20261019_V1/arm/axis`を提案する。まだ生成していない。
A/B・correctness・baseline/instrumentedは同じ各arm/axis prefixを再生し、X/Yは分離する。
H=160ごとに元の256-bit digest消費／余剰bit破棄を維持し、bit poolingへ変更しない。
旧128診断seed・旧消費済みclaimを再利用しない。これは固定workloadの決定論的比較でありIID証明ではない。
全Nの精度／母期待CPU費用のconfidenceをこのtraceから主張しない。seed規則の採択と新one-shot承認が必要。

mandatory STOP。
