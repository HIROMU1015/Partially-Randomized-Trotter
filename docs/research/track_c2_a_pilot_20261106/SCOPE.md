# C2限定pilot実行範囲（2026-11-06 JST）

基準6f6c084b、科学資産720416c5、同梱SCIENTIFIC_REVIEW.mdを使用。
ユーザーはこのC2-A/B限定pilotの最小実装・科学計算・合成・compile・専用branch公開を明示承認済み。
新しいgrant/承認証明は作らない。旧研究のSTOP・SOURCE・raw等は変更しない。
SERVER 2、主agentのみ。A/Bは別SOURCE・別出力・独立process。

共通：Linear H4、1.00 Å、STO-3G、DF12、保存元H ground、8 qubit/4 electron。
状態準備は除外。delta0=.02、m=0..17、t=.02*2^m、energy精度1.5936e-4 Ha、
alpha総和.05、各round位相総幅.4 rad。元H表現の無条件包含は未資格のまま。
E0は小系診断にのみ使用し、biasを補正せず、decoderへE0を渡さない。
保存common.pkl/eigenspectra、既存保証/alias/区間評価器を再利用し、SCF/DF/ground solveをしない。
RZ/CX/T/depthは別単位。新しい角度は局所的に照合し、旧compiler保証を無条件には移さない。

C2-A：固定splitごとに独立比較（LD3 primary、LD12 deterministic reference）。
S2/Y4、h=.02/.01、K=2、LD3のr=1,2,4,8,16,32が有限候補。
合成は全候補で固定の理想native policy、主指標RZとCX。
最初はm=0,8,17の最大144信号セル。PF選択前に同じ族内のh/rを最適化。
候補と固定PF対照の双方に同じ.4残余統計予算、uniform/cost-sensitive alpha機会を与える。
最小full-wrapper価格はq=1,2,4の決定論3点とLD3のorder0/order2全support局所界を使用。
長qは未資格proxyならそのまま表示し、外挿点値で勝者を認定しない。
primaryで有望なPF族交差がなければ18round探索・追加大量compileはせずSTOP。
交差があっても価格不確かさで未決ならSTOP。split切替をPF次数改善と混同しない。

承認された有限範囲の完了・保証不成立・科学的非改善・価格資格不足の時点でmandatory STOP。
技術debugは同じ条件で旧出力を保全して行う。自動grid拡大・C2-C・旧研究再開は禁止。
利用可能CPU/RAMを確認し、独立task数と実際の空き資源に応じてprocess並列。
共有設定、共有package、他者job/ファイルは変更しない。

実装時の限定価格診断：各task・初期3roundでsignal/primitive surrogateが選ぶrに限り、
q1/2/4、order0/rareの各stratumで決定的seedの代表pathを1本ずつcompileする。
最大144短価格セル。これは期待価格の推定/資格ではなく、価格不確かさを示す最小接続。
この代表pathから長qで科学的勝者を認定しない。全r compiled optimumも認定しない。
