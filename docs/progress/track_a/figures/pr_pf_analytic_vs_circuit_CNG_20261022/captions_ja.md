# 日本語caption・notes

固定条件・cohort・MC限界は[README](README.md)を参照。

## ε=0.05

図C：PR S2と固定PF S2のcompiled/native理論回路単価比。左右はcosine/sine、基準1。shot重みを混ぜず、各軸のRZ単価を比較する。PRの誤差棒は名目95% MC区間。

図N：保存済みHoeffding十分shot数を軸別に示す。独立N_circuitは未取得で、比は描かない。実機で必要shot数を測定した結果ではない。

図G：G00=ΣN_theory C_theoryを1とし、G01=ΣN_theory C_circuitを示す。N固定・Cのみ置換した総費用比較で、G11は未取得。誤差棒はpaired軸の共分散を含む古典MC平均の名目95%区間。

- H2 PR、rank3、L_D=1、q=1、δ=0.8、r=2、K=4、MC n=8。実施済み技術検査PASS範囲。
- H2 PF、rank3、L_D=3、q=1、δ=0.8、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H4 PR、rank10、L_D=3、q=1、δ=0.8、r=4、K=2、MC n=276。実施済み技術検査PASS範囲。
- H4 PF、rank10、L_D=5、q=1、δ=0.8、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H6 PR、rank19、L_D=4、q=2、δ=0.4、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H6 PF、rank19、L_D=7、q=1、δ=0.8、r=None、K=None、MC n=1。補助G不合格の条件付き。
- H8 PR、rank28、L_D=8、q=2、δ=0.4、r=4、K=4、MC n=128。補助G不合格の条件付き。
- H8 PF、rank28、L_D=11、q=1、δ=0.8、r=None、K=None、MC n=1。補助G不合格の条件付き。

## ε=0.01

図C：PR S2と固定PF S2のcompiled/native理論回路単価比。左右はcosine/sine、基準1。shot重みを混ぜず、各軸のRZ単価を比較する。PRの誤差棒は名目95% MC区間。

図N：保存済みHoeffding十分shot数を軸別に示す。独立N_circuitは未取得で、比は描かない。実機で必要shot数を測定した結果ではない。

図G：G00=ΣN_theory C_theoryを1とし、G01=ΣN_theory C_circuitを示す。N固定・Cのみ置換した総費用比較で、G11は未取得。誤差棒はpaired軸の共分散を含む古典MC平均の名目95%区間。

- H2 PR、rank3、L_D=1、q=1、δ=0.8、r=2、K=4、MC n=8。実施済み技術検査PASS範囲。
- H2 PF、rank3、L_D=3、q=1、δ=0.8、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H4 PR、rank10、L_D=3、q=2、δ=0.4、r=4、K=2、MC n=143。実施済み技術検査PASS範囲。
- H4 PF、rank10、L_D=7、q=2、δ=0.4、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H6 PR、rank19、L_D=4、q=4、δ=0.2、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H6 PF、rank19、L_D=11、q=2、δ=0.4、r=None、K=None、MC n=1。補助G不合格の条件付き。
- H8 PR、rank28、L_D=8、q=4、δ=0.2、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H8 PF、rank28、L_D=15、q=4、δ=0.2、r=None、K=None、MC n=1。補助G不合格の条件付き。

## ε=0.005

図C：PR S2と固定PF S2のcompiled/native理論回路単価比。左右はcosine/sine、基準1。shot重みを混ぜず、各軸のRZ単価を比較する。PRの誤差棒は名目95% MC区間。

図N：保存済みHoeffding十分shot数を軸別に示す。独立N_circuitは未取得で、比は描かない。実機で必要shot数を測定した結果ではない。

図G：G00=ΣN_theory C_theoryを1とし、G01=ΣN_theory C_circuitを示す。N固定・Cのみ置換した総費用比較で、G11は未取得。誤差棒はpaired軸の共分散を含む古典MC平均の名目95%区間。

- H2 PR、rank3、L_D=1、q=1、δ=0.8、r=2、K=4、MC n=8。実施済み技術検査PASS範囲。
- H2 PF、rank3、L_D=3、q=2、δ=0.4、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H4 PR、rank10、L_D=2、q=4、δ=0.2、r=4、K=2、MC n=128。実施済み技術検査PASS範囲。
- H4 PF、rank10、L_D=7、q=4、δ=0.2、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H6 PR、rank19、L_D=4、q=4、δ=0.2、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H6 PF、rank19、L_D=11、q=4、δ=0.2、r=None、K=None、MC n=1。補助G不合格の条件付き。
- H8 PR、rank28、L_D=8、q=4、δ=0.2、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H8 PF、rank28、L_D=15、q=4、δ=0.2、r=None、K=None、MC n=1。補助G不合格の条件付き。

## ε=0.001

図C：PR S2と固定PF S2のcompiled/native理論回路単価比。左右はcosine/sine、基準1。shot重みを混ぜず、各軸のRZ単価を比較する。PRの誤差棒は名目95% MC区間。

図N：保存済みHoeffding十分shot数を軸別に示す。独立N_circuitは未取得で、比は描かない。実機で必要shot数を測定した結果ではない。

図G：G00=ΣN_theory C_theoryを1とし、G01=ΣN_theory C_circuitを示す。N固定・Cのみ置換した総費用比較で、G11は未取得。誤差棒はpaired軸の共分散を含む古典MC平均の名目95%区間。

- H2 PR、rank3、L_D=1、q=4、δ=0.2、r=2、K=4、MC n=8。実施済み技術検査PASS範囲。
- H2 PF、rank3、L_D=3、q=4、δ=0.2、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H4 PR、rank10、L_D=2、q=8、δ=0.1、r=2、K=4、MC n=128。実施済み技術検査PASS範囲。
- H4 PF、rank10、L_D=7、q=8、δ=0.1、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H6 PR、rank19、L_D=4、q=8、δ=0.1、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H6 PF、rank19、L_D=11、q=8、δ=0.1、r=None、K=None、MC n=1。補助G不合格の条件付き。
- H8 PR、rank28、L_D=6、q=8、δ=0.1、r=4、K=4、MC n=128。補助G不合格の条件付き。
- H8 PF、rank28、L_D=15、q=8、δ=0.1、r=None、K=None、MC n=1。補助G不合格の条件付き。

## ε=0.0005

図C：PR S2と固定PF S2のcompiled/native理論回路単価比。左右はcosine/sine、基準1。shot重みを混ぜず、各軸のRZ単価を比較する。PRの誤差棒は名目95% MC区間。

図N：保存済みHoeffding十分shot数を軸別に示す。独立N_circuitは未取得で、比は描かない。実機で必要shot数を測定した結果ではない。

図G：G00=ΣN_theory C_theoryを1とし、G01=ΣN_theory C_circuitを示す。N固定・Cのみ置換した総費用比較で、G11は未取得。誤差棒はpaired軸の共分散を含む古典MC平均の名目95%区間。

- H2 PR、rank3、L_D=1、q=4、δ=0.2、r=2、K=4、MC n=8。実施済み技術検査PASS範囲。
- H2 PF、rank3、L_D=3、q=4、δ=0.2、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H4 PR、rank10、L_D=2、q=8、δ=0.1、r=2、K=4、MC n=128。実施済み技術検査PASS範囲。
- H4 PF、rank10、L_D=7、q=8、δ=0.1、r=None、K=None、MC n=1。実施済み技術検査PASS範囲。
- H6 PR、rank19、L_D=4、q=8、δ=0.1、r=4、K=2、MC n=128。補助G不合格の条件付き。
- H6 PF、rank19、L_D=11、q=8、δ=0.1、r=None、K=None、MC n=1。補助G不合格の条件付き。
- H8 PR、rank28、L_D=7、q=16、δ=0.05、r=4、K=4、MC n=128。補助G不合格の条件付き。
- H8 PF、rank28、L_D=15、q=8、δ=0.1、r=None、K=None、MC n=1。補助G不合格の条件付き。
