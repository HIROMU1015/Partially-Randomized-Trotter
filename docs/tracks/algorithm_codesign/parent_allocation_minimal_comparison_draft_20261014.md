# Parent配分：最小機構比較の草案

2026-10-14 JST。**草案であり、研究上の採択・実行契約・実行承認ではない。source準備／登録実行とも未開始、mandatory STOP。** [技術監査](parent_allocation_technical_audit_20261014.md)を受け、少数のroot変更で判別できる範囲を具体化する。今回、下記の登録係数・角度・資源表を計算していない。

## 判別する問いと候補domain

同じfinite P5を表すordinaryとclosed P5について、同じroot配分自由度と適用可能なproposal最適化を与え、元pairingに対する費用差と、return集約に固有の追加差を分ける。CTS・SPS全体への優位や最適配分をこの最小比較だけで認定しない。

候補入力は既存D1の既知3-qubit provider、p=(1/5,3/10,1/2)、sigma=+1、state000、tau=x=5/7、P5、r=1のみ。Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。I1を両方式へ同じように公開する。分子・geometry・basis・DF rank・split L_Dは適用外。この選択は事前案であり、新しい入力の取得やr最適化の認可ではない。

## 少数の候補と公平な比較

各表現のroot a,bを使い、次のL+1配分だけを結果前に候補とする。

- 元点 z_i=a b_i/s。
- 全parent質量を一childへ与えるL頂点 z=a e_j（全j）。

ordinaryのrootではa=1,b_i=x p_i。closed P5では既存a0/child0の式を使用する。両者の非root係数・provider・基礎targetを保持し、角度catalogueやseedの探索をしない。全jを与えるため、結果を見て安いchildだけを選ばない。

L=3なら二表現×四配分=8構成。各構成に以下の同じproposal機会を与える案とする。

| proposal | 扱い |
| --- | --- |
| canonical | digital係数比例。すべての構成に与える。 |
| regularized cost-aware IS | q∝c/sqrt(C_T+lambda)、lambda=1 Tの固定sampling用正則化案。lambdaを実費用へ加算しない。全構成へ同じ候補を与える。 |
| analytic cost-aware IS | 全positive-coefficient eventでC_T>0の場合のみq∝c/sqrt(C_T)も与える。零T costの場合は未達成極限を出力lawにせず、その理由を明示する。 |

最大24 lawを同じ有限confidence policyで会計する。これは有限proposal候補の比較であり、range・ceil付き全proposal最適性ではない。零T eventでの正則化とlambdaは**実行前に採択すべき比較仕様**であり、この草案が自動採択しない。旧方式にもanalytic ISを与えるため、新側canonical対旧側canonicalだけを強い対照への利益と呼ばない。配分を新側だけへ与えてRA利益とも呼ばない。

global IS用のevent費用情報は固定D1保存表・IRを両表現へ同じように公開する案。新角度以外の価格はidentity確認付きで再利用する。この表を生成した過去の費用、表走査、proposal初期化とonline生成の扱いを別欄に残す。表利用の小規模I1比較から、一般involution I0でglobal ISが安価に生成できると推論しない。

独立したCTS/SPSの追加最適化をこの最小runへ混ぜない。既存matched CTSの保存値を説明用に参照する場合も、新ledgerでの競争結果として流用しない。強い対照の追加が必要かは結果と主張に応じてGPTで判断する。

## finite-bit・精度・confidenceの案

root/probability bitsはK256/H160、rho=eta=1e-12、primitive synthesis precision=1e-6を既存設定から引き継ぐ案。新zはexact rational、root c_iは区間とmidpoint、qはdyadic、w=tilde c/qとする。分布ごとにmoment/range/normalizerを検証する。

axis epsilonは1/200。最大24 lawのX/Y48 axesへfamily alpha=0.05を均等配分し、axis alpha=1/960とする案。native取得の失敗は成功・negativeに数えずtechnical outcomeとし、旧runのalphaや分類を変更しない。この新ledgerの十分Nは将来の実行認可後にだけ生成する。旧D1/D3のNをこの新alphaで再採点した記録を上書きしない。

Taylor保証はHermitian contractionの|x|^6/6!を全構成で共通に使う。係数・atan解像・strict synthesis誤差を別々に戻し、元の保守的axis変換を維持する。実際に残る統計幅を共通式で使用し、未使用分の扱いを結果前に固定する。synthesis精度を上げた追加余裕は与えない。実行中のbits/tolerance/precision救済は禁止する。

必要N、range、biasが上限を超えたlawは不適格と明示し、未完了prefixやmissing priceを勝敗へ用いない。Nは実測shotsでも必要最小shotsでもない。shot cap候補は既存と同じ1e9/axis。

## 新規取得を小さくする方法

頂点jでは、そのchildだけratio=b_j/aを持ち、他のchildはpure-Q endpointとなる。新しいpositive ratioの静的上限は二表現×L、**L=3なら最大6 unique keys**。元点と非rootは既存cache identityが一致する場合のみ再利用する。新ratioに旧価格を代理しない。endpointはstrict phaseを保持したexact回路とし、巨大ratioとして合成しない。

将来sourceではroot native IRを最大24 event分だけ構成し、共通のliteral簡約・加法的価格を用いる案。native transpile/whole-circuit compilerは0。新sequence内部やstep間の相殺へ進まず、r1なのでD2の219,303 pairと旧節約額を使用しない。全finite-target行列の再評価は不要とする案だが、新adapterのcontrolled意味論はまず登録外で検証する必要がある。実registered matrix検証を追加する場合は範囲を再固定して認可を受ける。

合成tool/version/seed/strict guardは既存D1固定runtimeと同じidentityへbindし、実行前source manifestにwheel/package/source hashesを明記する。tool未確認のままworkerを起動しない。

| 資源・失敗処理の事前案 | 上限／規則 |
| --- | --- |
| 新unique synthesis keys / calls | 最大6、各一回、retry0。必要keyが超えたら停止し、探索で削らない。 |
| worker / numeric threads | 1 / 1。少数keyなのでcache共有とaggregate memoryを優先。 |
| RSS / artifact / log | aggregate512 MiB / 32 MiB / 4 MiB。現時点で適合実測はしていない。 |
| CPU | total600 s、per-key90 sという技術案。実行前の性能確認で仕様化する。 |
| phase / total wall | 新規契約の共通方針に従い、wall上限を設けない。tool polling timeoutとは分離。 |
| missing/error/cap/certificate failure | retryせずtechnical inconclusive。native未取得値を0、成功prefixを最終結果にしない。 |
| source / marker | 新sourceと別authorization-only直接子、新namespace/exclusive marker。旧markerを再利用しない。 |

## 保存・報告するもの

配分、係数区間、angle/endpoint kind、実proposal/weight/moment/range、実装bias、十分N、key identityとT/CX/1Q費用を対応付ける。主費用は各指標のG(h)=2N(EC+h)。Tをnative1Qへ重複加算しない。h=0を準備無料の測定結果とはせず、準備費用を記号変数として残す。

元common-angleとの固定表現内の差、ordinary側にも同じ変更を与えた差、RAと非RAの差を分離する。native価格が未取得なら固定単価proxyだけをreportし、native優位をclassificationしない。classical setup/online sampling/新angle取得費用も別欄とし、未知online費用を0としない。

## この草案から先へ進む条件

今回閉じたのは数式と実装接続の監査であり、上記のinput・proposal候補・confidence配分・resource capsは**比較計画案**である。新source、finite-bit/endpoint adapter、strict phase guard、focused native tests、launch manifestの実装は別の技術準備指示に従う。

正則化proposalの比較上の意味、zero-cost処理、有限候補の主張範囲を結果前に確定できない場合は、その論点をGPTへ返す。技術準備完了だけで登録計算を開始せず、ユーザーの別途明示実行承認を必要とする。終了後は結果によらずmandatory STOPし、科学的優劣・継続・新規性の判断はGPTへ戻す。
