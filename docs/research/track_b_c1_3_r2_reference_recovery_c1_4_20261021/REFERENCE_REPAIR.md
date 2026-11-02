# Reference復元：原因確定と実行前修正

基点は`49dae5247ff0ad2c4efb9056e095b70913cc9ddb`。旧C1-3/C1-3-R1の失敗、SOURCE、authorization、STOPは保持し、新namespaceへ追加する。共通運用指示2026-10-17（特に§8/8.4）とprojects/AGENTS.mdを適用し、主agentのみで作業する。

旧`reference.saved_basis`は保存real/imag hexを`real + 1j*imag`でcomplex128へ組み立てた。新`bitwise.restore_local`は`<c16`のC-order配列へ実部・虚部を別々に直接格納する。保存値・期待SHA・basis ID/hashを変更せず、raw bytes照合を保持する。新referenceは保存hexから構成し、adapterのoperator出力をexpectedに使わない。新referenceの他の関数は旧SOURCEとAST同一である。

Phase Aは正負zeroの全実虚組合せ、正負有限値、subnormal、最大有限値で、hex→binary64→complex128→C-orderを独立な`struct.pack(<dd)`のbytesと比較した。3局所行列fixtureすべてexact一致。旧方式との差分byte、failure/logger/terminal、既存10人工controlled比較・63-event first momentのPASSを保存した。

Phase BはR1の13 basis・412保存局所行列を各1回復元した。新方式は**412件すべて保存SHAおよび独立packed bytesと一致**、descriptor順・basis hashも維持。旧方式は59行列で不一致となり、すべての相違binary64成分はzeroの符号bitだけだった。

前回と同じselected basis順で最初に不一致となるのは`basis_002.json`、basis ID `df-basis-48e9eb82b09846e1f479a92da16fb891cf5d5d0c80dbfffbe852bbab35a09108`、operation index 0、RZGate。保存hashは`0fdea6ca3d6006053c50e6cc39c328b15c3011a02670abc93def3ed1ebadfeab`、旧復元hashは`fbfcd8f81e798f2e4e04256375c0766e6166910440ffd3de2b9b37a88ab5aac7`。最初の相違byte offsetは15。offset 8からの虚部binary64は`0x8000000000000000`（-0）から`0x0`（+0）へ変わる。詳細なexpected/actual bytesと元hexは[legacy_first_difference.json](phase_b/legacy_first_difference.json)にある。

これは人工fixtureからの推測ではなく、固定R1保存局所行列・同じruntime・旧式の再現に基づく。前回run自体の記録を追加・改変したものではない。符号zeroの正規化で受理したのではなく、元の保存bytesへexactに復元した。

Phase Bでは局所matrixの同一性確認のみを行い、full basis/operator/event query/X/Y/NPZ再解析/native/samplingは行っていない。Phase Cでは別途、新SOURCE/contract/authorizationを固定した独立one-shotで6 eventを検証する。412局所matrixのbitwise一致を、全label/event/pathの数値correctnessやnative保証と同一視しない。

Phase Cの科学条件・6 event・精度・1e-10のcontrolled operator基準・reference数学対象・norm・phaseは旧契約と同一。角度とrootは既存F3、state hash/journalは固定R1修正版を再利用し、sampler/adapterを変更しない。科学実行後のSOURCE修正・再試行は禁止。PASS時だけC1-4の静的契約整備へ進み、その後は公開・保全確認してmandatory STOPする。
