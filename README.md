MGPWeka
=======

Implements Matrix Multiplication using Multicore and GPU approaches.

Dataset from UCI Repository: http://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis

- GPU Implementation -> Using JCublas library
trunk/weka-dev/src/main/java/weka/core/matrix/JCublasMatrixMultiplication.java

- Multicore Implementation
trunk/weka-dev/src/main/java/weka/core/matrix/MulticoreMatrixMult.java

- Method called on
trunk/weka-dev/src/main/java/weka/core/matrix/Matrix.java ->method times(Matrix B)

Authors:

tengel@inf.ufsm.br - Universidade Federal de Santa Maria - Brazil

andrea@inf.ufsm.br - Universidade Federal de Santa Maria - Brazil

Luiz-Angelo.Steffenel@univ-reims.fr - Université de Reims Champagne-Ardenne - France

Manuele.Kirsch-Pinheiro@univ-paris1.fr - Université Paris 1 Panthéon-Sorbonne - France
