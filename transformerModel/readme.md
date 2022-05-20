**有关transformer的疑问**

transformer中Q和K实际上维度是有差别的，Q(L,d)，K(S,d)，Q*K的维度为(L,S)，实际上L<S，因为L目标序列的长度可以认为是序列的实际长度，而S则是所有训练样本中最长的序列长度。（错误）

transformer对于padding后的query可以有效获取对key的attention，但是有效的query无法有效获取padding后的key的attention。

