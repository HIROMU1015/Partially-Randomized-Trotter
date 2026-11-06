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

C2-B：PFはFull Y4/LD12/h=.02の保存18round固定（tail-free endpoint）。
隣接する同一fragmentを融合し、同じbasis/center/actual inverse実装を全armに与える。
中心のcontrolled scalarは保持し、帰りは合成済みbasisの実adjointを使う。
中心角度とfull-space fragment幅Wを評価し、min(2,abs(tau)*W)をbasis誤差重みに使用。
通常加算の一様精度、既知の離散非一様配分（angle key別の1 decade relaxation）、
構造上界の一様/非一様配分を比較する。新しいresidual/normalizationは導入しない。
共通の合成phase cap .01/.02 radを有限感度候補とし、各armに同じ選択機会を与える。
既存acquire/strict_guard/actual-angle seed17を再利用。必要なnative angle/precision keyのみ合成。
実wordのT数を使用し、全18roundのoperator disk→phase/radius→shot→総Tに接続。
局所full-space operator検査、実adjoint、D=I cancellation、strict guardを検証する。
全round回路は周期SOURCEの厳密なblock多重度で表現し、大規模unrollを避ける。
全空間の合成誤差上界を使用し、合成後basisを70次元sectorへ射影しない。

承認された有限範囲の完了・保証不成立・科学的非改善・価格資格不足の時点でmandatory STOP。
技術debugは同じ条件で旧出力を保全して行う。自動grid拡大・C2-C・旧研究再開は禁止。
利用可能CPU/RAMを確認し、独立task数と実際の空き資源に応じてprocess並列。
共有設定、共有package、他者job/ファイルは変更しない。
