??<
?0?/
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
<
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??6
?&
ConstConst*
_output_shapes	
:?*
dtype0	*?%
value?%B?%	?"?%                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      
?"
Const_1Const*
_output_shapes	
:?*
dtype0*?!
value?!B?!?BleBdeBpageBenBs’B
exemplaireBfoisBsurBappelerBnommerBàBunB	intitulerB
impressionBl’BmoiBjeBfichierBmerciBcelaBpourBimprimerBtirageBtaireBéditeB	dénommerBd’BprojetBj’BpdfBseBimprimeBpossibleBsiBêtreBcoucouBcoolBteBquiBcontenirBplzBhelloBimageBcopieBjusqu’BtirBdocumentBtoiBmêmeB	imprimantBéditerBtireBlanceBéditionBavanceBvouloirB	souhaiterBilBstpBphotoBtirerBest-ilBpasBneB	dérangerBbesoinBauraiBthanksBarticleBaimeraiBsortirBfileBaimerBplerBmeBbonjourBt’B	remercierBm’BsalutBposteBbalanceBenvoyerBsendB
transfèreBnameBimprimB	commencerBattenteBfaireBpeuxBpouvoirBsauraiBsaiBpeuBdessinBprésentationBmonB	comlimentBavecB	dénomméBinviteBtyBplusBversoBrectoBrapportBcopierBeffectueBheyBplopBpleaseBmrcBdansBsiedBvidéoBbyB	démarrerBstarterBbyeBlancerBcodeBpapierBversionBbalancerBmettreBdémarrBdémarreBattendreBfeuilletBfeuilleB
intitulantBnomméBlyyBbrbiBisdiimBaadspBnfbxyBxxqrBmxxnhjvmBkeoygBjsguuBvwicbBvhvwnBpqhemBgwbmBdxlwdBbwypBzagtqxcBvuonpBupxawtBtavbhcBqllhBwwxxufBraivoqBoajhawBmahvtuBezidyBxtcsBnxggBnfteBkteqrwBfxwwiBbnlgBazffdvBatdbqBrlwtuzBqvrdcBnebnBjooerBzhekbxBvnnhwpButrywkjBuogqhrfhBfuximuyBdzhmByqtqfafBvqfpBstamBoykvvjBocwrdrzBkioqauBjiqbqfBesdfdqBcyjfoBbxejBzbllaakBxmqvxaBtyogBseujBqyltxsBimlnhnBarabByqpBrcotcqBknrokzBftgbdcfBeqmmchBelqkioBebkhBdrjzflBdjnosdeBzwigaBqogqoqBevskBasmkBthodhzBoegzBlfzpBjisopBggbkqBdplosbfmBdmxkcpBaomhBzyfegnBydqqBibnkjvBhbbpBdjtfwBxmumqaBhhmeBslzwBrmeklnwBafxshBncvxywBcjbwyBsfqvBapfamBqstcoB	intituléBçaBnommantBwonoBpjikgpBoksgBwlntoBhedkosmBdcpkvcyBxzanBxvvzxwBxjlpayBwwndcwBwvmwbwBwpxmBwgwqeBtzigbBrlzuBqaocbkBjqhwBekenwBatpfzcBznydwjkBwduqBrmstBqlxvBoavjaqBmrjhBcjdxfBccorBznudgBzjjxpvByrnyqyBvmcclBtysorkBtmwsdhBtgthtBsulbBrhkvcqBoxjigjtBerpazmiBcaakbbbBbvkgrbszBapxirByxrlByndpuByjicxBxxwpBxuspBtgcbBsnsskdBqvzoqsBnnxBnhwhuBmveenyBjklrBjeazdBihiqBihduBhvdxBgxltBbgfcclnBahwidrBadojBzprhByyquByvfyBytqvBxwrimBxjvdBwvpzBwquhuBuwqzbBthrnjBrxaugevBrcartBqnwlBoaqztBmdpvBjcgajczfBioreBinozBikvmjgxuBhiyjBhiqdBeitfBduhkBchbhBbhnwByembBybxmByardlBwxrkolBvrjttpBtqkxBrjqpBqyzqvnBondfBomecffBoitcwkBnyohpoerBnaxvBmgwytBlobqynbBkxfrBjryuhaBitgafBiikoBigdvBgxlqvsmBdjhqiBbzgvpBbqfBviuvxgBtukkotvBrgekyBqrtpzyBqebmhweBmlahkhBkrteBjtenhzBjsieBfsqaBecxwvBcgzaBbyeqlcBbvatBaqjvjrBziebvakByotdnBxedbBwebktwBvtvuBvrwkgnBvnwjyBoqzpfzBoedqjBnedwBmowkkxyBldkBjbnoBholbBfzfcvBzygxxperBxiarvtiBtnotpBosunjBkmguBixpfBgytwkuzBbwxcrBavihlBzxcsoBzmzsiBwsnxBvisxBtethsrBsmduvBpsksfzBmgfbBkjothBjbksajBhvkhheyoBhpxfBeksrwuwlBdqhxBczwjBcfoqBbxreBbjqbBzwlnnplBzmjqfhByvtznfBxlyngdtBwrjdiiBvtuqBvlwjBttakBsxrpvcBspcablBpletoypBpbdlxBoqgkkBmlmwBkcsthqBjcwcBhrfdkBbqqfBzrebBznmwjByidlkBwyhlrBwimatoaBwborbBvxqlnBujuhBscmkBrpmcBpvtsftBoblpBnmgcBlzqrtbBjsdfuBikdwmBienntkBgqidvBeojftBddgdwBcrcouBbzuwstBzupdzmByaghBxzerfBxomvBxmjkhcrBtqyybekpBtgotkBsggfBpvmociBlvpdBjuqtBhxrcfwjBhlqerBhedxxkBgwwlqBghqaobBgasBfogxhigcBfbygvBbuoseBacjvkBwghvmotBvefjnBvdhfbBtrpeBtpmhfveBsgqwfhpBrppwfnBrozukBqiybfdmBpjbeBownomvBomocBmebeBlhxhBirngBghzkcBelhhB	bpbcvurdbBbbpskByzguBvonhxmwfBnhfwBmddfBmctcBfjimgbBdocmkyBwfdqcBrvpuBtbmBsstoBsjldBpejfBpaclBoituBmwckBhemfBdbskBzznnBzzgwbBzyrbcBzybmqBzxtjtlBzwqrBztmyepvBzschBzrnaBzpkhBzltbBzljrbiBzkuglBzjylrBzhymuBzhwtBzhbzdheBzfoozBzegomyB	zeftiuhegBzbqzBzbmsbBzaxjBzaqkexgBzahlByxnvbynBywypkBywwkByvtvuhByufnsvByszvByrogknB	yqcityrvyBypkvxbBynzfhBykqbnBykkqiuBykfmByjrqpByhpByhnotBygzgwByeiuBybzoBybyBxzycBxxutwnBxwuhikBxtoonwBxthbvyvBxtdrwonBxssgbBxrpBxplyrwBxpgufyzBxneyBxmmreBxlnvndBxiaBxhtulBxgjzBxdaaolBwzbkBwyhaBwxmogBwwvivBwusfBwsuzBwraBwomhBwnjugBwlsxBwliqmBwlbqdBwhyvBweydnzBwbaaydkBvxxhxcmBvxvuBvwwuBvvnnBvtbmpBvsafBvryqrBvqgefgBvmwhnBvlklBvkzbBvjivkdBvjbabBvhsbuBvhjwfbBvfjlBveuujBvbrwBvacuBuyizggBuxdjBuvafeBuuqjtjvBusciq
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_5Const*
_output_shapes
: *
dtype0*
valueB B 
?
*Adam/transformer_decoder_23/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/transformer_decoder_23/dense_1/bias/v
?
>Adam/transformer_decoder_23/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_23/dense_1/bias/v*
_output_shapes
:*
dtype0
?
,Adam/transformer_decoder_23/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/transformer_decoder_23/dense_1/kernel/v
?
@Adam/transformer_decoder_23/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/transformer_decoder_23/dense_1/kernel/v*
_output_shapes

: *
dtype0
?
(Adam/transformer_decoder_23/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/transformer_decoder_23/dense/bias/v
?
<Adam/transformer_decoder_23/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/transformer_decoder_23/dense/bias/v*
_output_shapes
: *
dtype0
?
*Adam/transformer_decoder_23/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adam/transformer_decoder_23/dense/kernel/v
?
>Adam/transformer_decoder_23/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_23/dense/kernel/v*
_output_shapes

: *
dtype0
?
8Adam/transformer_decoder_23/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_decoder_23/layer_normalization_1/beta/v
?
LAdam/transformer_decoder_23/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_23/layer_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
9Adam/transformer_decoder_23/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/transformer_decoder_23/layer_normalization_1/gamma/v
?
MAdam/transformer_decoder_23/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/transformer_decoder_23/layer_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_decoder_23/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_decoder_23/layer_normalization/beta/v
?
JAdam/transformer_decoder_23/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_23/layer_normalization/beta/v*
_output_shapes
:*
dtype0
?
7Adam/transformer_decoder_23/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_decoder_23/layer_normalization/gamma/v
?
KAdam/transformer_decoder_23/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_decoder_23/layer_normalization/gamma/v*
_output_shapes
:*
dtype0
?
HAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/v
?
\Adam/transformer_decoder_23/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0
?
JAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/v
?
^Adam/transformer_decoder_23/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
?
=Adam/transformer_decoder_23/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_23/multi_head_attention/value/bias/v
?
QAdam/transformer_decoder_23/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_23/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_23/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_decoder_23/multi_head_attention/value/kernel/v
?
SAdam/transformer_decoder_23/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_23/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
?
;Adam/transformer_decoder_23/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_decoder_23/multi_head_attention/key/bias/v
?
OAdam/transformer_decoder_23/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp;Adam/transformer_decoder_23/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
?
=Adam/transformer_decoder_23/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_decoder_23/multi_head_attention/key/kernel/v
?
QAdam/transformer_decoder_23/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_23/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
?
=Adam/transformer_decoder_23/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_23/multi_head_attention/query/bias/v
?
QAdam/transformer_decoder_23/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_23/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_23/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_decoder_23/multi_head_attention/query/kernel/v
?
SAdam/transformer_decoder_23/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_23/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
?
*Adam/transformer_encoder_23/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/transformer_encoder_23/dense_1/bias/v
?
>Adam/transformer_encoder_23/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_23/dense_1/bias/v*
_output_shapes
:*
dtype0
?
,Adam/transformer_encoder_23/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/transformer_encoder_23/dense_1/kernel/v
?
@Adam/transformer_encoder_23/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/transformer_encoder_23/dense_1/kernel/v*
_output_shapes

: *
dtype0
?
(Adam/transformer_encoder_23/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/transformer_encoder_23/dense/bias/v
?
<Adam/transformer_encoder_23/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/transformer_encoder_23/dense/bias/v*
_output_shapes
: *
dtype0
?
*Adam/transformer_encoder_23/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adam/transformer_encoder_23/dense/kernel/v
?
>Adam/transformer_encoder_23/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_23/dense/kernel/v*
_output_shapes

: *
dtype0
?
8Adam/transformer_encoder_23/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_encoder_23/layer_normalization_1/beta/v
?
LAdam/transformer_encoder_23/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_encoder_23/layer_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
9Adam/transformer_encoder_23/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/transformer_encoder_23/layer_normalization_1/gamma/v
?
MAdam/transformer_encoder_23/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/transformer_encoder_23/layer_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_encoder_23/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_encoder_23/layer_normalization/beta/v
?
JAdam/transformer_encoder_23/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_encoder_23/layer_normalization/beta/v*
_output_shapes
:*
dtype0
?
7Adam/transformer_encoder_23/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_encoder_23/layer_normalization/gamma/v
?
KAdam/transformer_encoder_23/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_encoder_23/layer_normalization/gamma/v*
_output_shapes
:*
dtype0
?
HAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/v
?
\Adam/transformer_encoder_23/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0
?
JAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/v
?
^Adam/transformer_encoder_23/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
?
=Adam/transformer_encoder_23/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_23/multi_head_attention/value/bias/v
?
QAdam/transformer_encoder_23/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_23/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_23/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_encoder_23/multi_head_attention/value/kernel/v
?
SAdam/transformer_encoder_23/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_23/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
?
;Adam/transformer_encoder_23/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_encoder_23/multi_head_attention/key/bias/v
?
OAdam/transformer_encoder_23/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp;Adam/transformer_encoder_23/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
?
=Adam/transformer_encoder_23/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_encoder_23/multi_head_attention/key/kernel/v
?
QAdam/transformer_encoder_23/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_23/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
?
=Adam/transformer_encoder_23/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_23/multi_head_attention/query/bias/v
?
QAdam/transformer_encoder_23/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_23/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_23/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_encoder_23/multi_head_attention/query/kernel/v
?
SAdam/transformer_encoder_23/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_23/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
?
FAdam/token_and_position_embedding_28/position_embedding29/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *W
shared_nameHFAdam/token_and_position_embedding_28/position_embedding29/embeddings/v
?
ZAdam/token_and_position_embedding_28/position_embedding29/embeddings/v/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_28/position_embedding29/embeddings/v*
_output_shapes

: *
dtype0
?
CAdam/token_and_position_embedding_28/token_embedding29/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*T
shared_nameECAdam/token_and_position_embedding_28/token_embedding29/embeddings/v
?
WAdam/token_and_position_embedding_28/token_embedding29/embeddings/v/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_28/token_embedding29/embeddings/v*
_output_shapes

:*
dtype0
?
FAdam/token_and_position_embedding_27/position_embedding28/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *W
shared_nameHFAdam/token_and_position_embedding_27/position_embedding28/embeddings/v
?
ZAdam/token_and_position_embedding_27/position_embedding28/embeddings/v/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_27/position_embedding28/embeddings/v*
_output_shapes

: *
dtype0
?
CAdam/token_and_position_embedding_27/token_embedding28/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*T
shared_nameECAdam/token_and_position_embedding_27/token_embedding28/embeddings/v
?
WAdam/token_and_position_embedding_27/token_embedding28/embeddings/v/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_27/token_embedding28/embeddings/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_24/kernel/v
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/v
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:@*
dtype0
?
*Adam/transformer_decoder_23/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/transformer_decoder_23/dense_1/bias/m
?
>Adam/transformer_decoder_23/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_23/dense_1/bias/m*
_output_shapes
:*
dtype0
?
,Adam/transformer_decoder_23/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/transformer_decoder_23/dense_1/kernel/m
?
@Adam/transformer_decoder_23/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/transformer_decoder_23/dense_1/kernel/m*
_output_shapes

: *
dtype0
?
(Adam/transformer_decoder_23/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/transformer_decoder_23/dense/bias/m
?
<Adam/transformer_decoder_23/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/transformer_decoder_23/dense/bias/m*
_output_shapes
: *
dtype0
?
*Adam/transformer_decoder_23/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adam/transformer_decoder_23/dense/kernel/m
?
>Adam/transformer_decoder_23/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_23/dense/kernel/m*
_output_shapes

: *
dtype0
?
8Adam/transformer_decoder_23/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_decoder_23/layer_normalization_1/beta/m
?
LAdam/transformer_decoder_23/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_23/layer_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
9Adam/transformer_decoder_23/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/transformer_decoder_23/layer_normalization_1/gamma/m
?
MAdam/transformer_decoder_23/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/transformer_decoder_23/layer_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_decoder_23/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_decoder_23/layer_normalization/beta/m
?
JAdam/transformer_decoder_23/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_23/layer_normalization/beta/m*
_output_shapes
:*
dtype0
?
7Adam/transformer_decoder_23/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_decoder_23/layer_normalization/gamma/m
?
KAdam/transformer_decoder_23/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_decoder_23/layer_normalization/gamma/m*
_output_shapes
:*
dtype0
?
HAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/m
?
\Adam/transformer_decoder_23/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0
?
JAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/m
?
^Adam/transformer_decoder_23/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
?
=Adam/transformer_decoder_23/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_23/multi_head_attention/value/bias/m
?
QAdam/transformer_decoder_23/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_23/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_23/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_decoder_23/multi_head_attention/value/kernel/m
?
SAdam/transformer_decoder_23/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_23/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
?
;Adam/transformer_decoder_23/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_decoder_23/multi_head_attention/key/bias/m
?
OAdam/transformer_decoder_23/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp;Adam/transformer_decoder_23/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
?
=Adam/transformer_decoder_23/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_decoder_23/multi_head_attention/key/kernel/m
?
QAdam/transformer_decoder_23/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_23/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
?
=Adam/transformer_decoder_23/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_23/multi_head_attention/query/bias/m
?
QAdam/transformer_decoder_23/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_23/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_23/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_decoder_23/multi_head_attention/query/kernel/m
?
SAdam/transformer_decoder_23/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_23/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
?
*Adam/transformer_encoder_23/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/transformer_encoder_23/dense_1/bias/m
?
>Adam/transformer_encoder_23/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_23/dense_1/bias/m*
_output_shapes
:*
dtype0
?
,Adam/transformer_encoder_23/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *=
shared_name.,Adam/transformer_encoder_23/dense_1/kernel/m
?
@Adam/transformer_encoder_23/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/transformer_encoder_23/dense_1/kernel/m*
_output_shapes

: *
dtype0
?
(Adam/transformer_encoder_23/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/transformer_encoder_23/dense/bias/m
?
<Adam/transformer_encoder_23/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/transformer_encoder_23/dense/bias/m*
_output_shapes
: *
dtype0
?
*Adam/transformer_encoder_23/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *;
shared_name,*Adam/transformer_encoder_23/dense/kernel/m
?
>Adam/transformer_encoder_23/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_23/dense/kernel/m*
_output_shapes

: *
dtype0
?
8Adam/transformer_encoder_23/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_encoder_23/layer_normalization_1/beta/m
?
LAdam/transformer_encoder_23/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_encoder_23/layer_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
9Adam/transformer_encoder_23/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/transformer_encoder_23/layer_normalization_1/gamma/m
?
MAdam/transformer_encoder_23/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/transformer_encoder_23/layer_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_encoder_23/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_encoder_23/layer_normalization/beta/m
?
JAdam/transformer_encoder_23/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_encoder_23/layer_normalization/beta/m*
_output_shapes
:*
dtype0
?
7Adam/transformer_encoder_23/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_encoder_23/layer_normalization/gamma/m
?
KAdam/transformer_encoder_23/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_encoder_23/layer_normalization/gamma/m*
_output_shapes
:*
dtype0
?
HAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Y
shared_nameJHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/m
?
\Adam/transformer_encoder_23/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0
?
JAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*[
shared_nameLJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/m
?
^Adam/transformer_encoder_23/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
?
=Adam/transformer_encoder_23/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_23/multi_head_attention/value/bias/m
?
QAdam/transformer_encoder_23/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_23/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_23/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_encoder_23/multi_head_attention/value/kernel/m
?
SAdam/transformer_encoder_23/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_23/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
?
;Adam/transformer_encoder_23/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_encoder_23/multi_head_attention/key/bias/m
?
OAdam/transformer_encoder_23/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp;Adam/transformer_encoder_23/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
?
=Adam/transformer_encoder_23/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*N
shared_name?=Adam/transformer_encoder_23/multi_head_attention/key/kernel/m
?
QAdam/transformer_encoder_23/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_23/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
?
=Adam/transformer_encoder_23/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_23/multi_head_attention/query/bias/m
?
QAdam/transformer_encoder_23/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_23/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_23/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?Adam/transformer_encoder_23/multi_head_attention/query/kernel/m
?
SAdam/transformer_encoder_23/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_23/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
?
FAdam/token_and_position_embedding_28/position_embedding29/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *W
shared_nameHFAdam/token_and_position_embedding_28/position_embedding29/embeddings/m
?
ZAdam/token_and_position_embedding_28/position_embedding29/embeddings/m/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_28/position_embedding29/embeddings/m*
_output_shapes

: *
dtype0
?
CAdam/token_and_position_embedding_28/token_embedding29/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*T
shared_nameECAdam/token_and_position_embedding_28/token_embedding29/embeddings/m
?
WAdam/token_and_position_embedding_28/token_embedding29/embeddings/m/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_28/token_embedding29/embeddings/m*
_output_shapes

:*
dtype0
?
FAdam/token_and_position_embedding_27/position_embedding28/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *W
shared_nameHFAdam/token_and_position_embedding_27/position_embedding28/embeddings/m
?
ZAdam/token_and_position_embedding_27/position_embedding28/embeddings/m/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_27/position_embedding28/embeddings/m*
_output_shapes

: *
dtype0
?
CAdam/token_and_position_embedding_27/token_embedding28/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*T
shared_nameECAdam/token_and_position_embedding_27/token_embedding28/embeddings/m
?
WAdam/token_and_position_embedding_27/token_embedding28/embeddings/m/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_27/token_embedding28/embeddings/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_24/kernel/m
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_23/kernel/m
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_548861*
value_dtype0	
n

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name614410*
value_dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
?
#transformer_decoder_23/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#transformer_decoder_23/dense_1/bias
?
7transformer_decoder_23/dense_1/bias/Read/ReadVariableOpReadVariableOp#transformer_decoder_23/dense_1/bias*
_output_shapes
:*
dtype0
?
%transformer_decoder_23/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%transformer_decoder_23/dense_1/kernel
?
9transformer_decoder_23/dense_1/kernel/Read/ReadVariableOpReadVariableOp%transformer_decoder_23/dense_1/kernel*
_output_shapes

: *
dtype0
?
!transformer_decoder_23/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!transformer_decoder_23/dense/bias
?
5transformer_decoder_23/dense/bias/Read/ReadVariableOpReadVariableOp!transformer_decoder_23/dense/bias*
_output_shapes
: *
dtype0
?
#transformer_decoder_23/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#transformer_decoder_23/dense/kernel
?
7transformer_decoder_23/dense/kernel/Read/ReadVariableOpReadVariableOp#transformer_decoder_23/dense/kernel*
_output_shapes

: *
dtype0
?
1transformer_decoder_23/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31transformer_decoder_23/layer_normalization_1/beta
?
Etransformer_decoder_23/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp1transformer_decoder_23/layer_normalization_1/beta*
_output_shapes
:*
dtype0
?
2transformer_decoder_23/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42transformer_decoder_23/layer_normalization_1/gamma
?
Ftransformer_decoder_23/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp2transformer_decoder_23/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
?
/transformer_decoder_23/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_decoder_23/layer_normalization/beta
?
Ctransformer_decoder_23/layer_normalization/beta/Read/ReadVariableOpReadVariableOp/transformer_decoder_23/layer_normalization/beta*
_output_shapes
:*
dtype0
?
0transformer_decoder_23/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_decoder_23/layer_normalization/gamma
?
Dtransformer_decoder_23/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp0transformer_decoder_23/layer_normalization/gamma*
_output_shapes
:*
dtype0
?
Atransformer_decoder_23/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAtransformer_decoder_23/multi_head_attention/attention_output/bias
?
Utransformer_decoder_23/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpAtransformer_decoder_23/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
?
Ctransformer_decoder_23/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECtransformer_decoder_23/multi_head_attention/attention_output/kernel
?
Wtransformer_decoder_23/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpCtransformer_decoder_23/multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
?
6transformer_decoder_23/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_decoder_23/multi_head_attention/value/bias
?
Jtransformer_decoder_23/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp6transformer_decoder_23/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
?
8transformer_decoder_23/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_decoder_23/multi_head_attention/value/kernel
?
Ltransformer_decoder_23/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp8transformer_decoder_23/multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0
?
4transformer_decoder_23/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64transformer_decoder_23/multi_head_attention/key/bias
?
Htransformer_decoder_23/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp4transformer_decoder_23/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
?
6transformer_decoder_23/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86transformer_decoder_23/multi_head_attention/key/kernel
?
Jtransformer_decoder_23/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp6transformer_decoder_23/multi_head_attention/key/kernel*"
_output_shapes
:*
dtype0
?
6transformer_decoder_23/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_decoder_23/multi_head_attention/query/bias
?
Jtransformer_decoder_23/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp6transformer_decoder_23/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
?
8transformer_decoder_23/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_decoder_23/multi_head_attention/query/kernel
?
Ltransformer_decoder_23/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp8transformer_decoder_23/multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0
?
#transformer_encoder_23/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#transformer_encoder_23/dense_1/bias
?
7transformer_encoder_23/dense_1/bias/Read/ReadVariableOpReadVariableOp#transformer_encoder_23/dense_1/bias*
_output_shapes
:*
dtype0
?
%transformer_encoder_23/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *6
shared_name'%transformer_encoder_23/dense_1/kernel
?
9transformer_encoder_23/dense_1/kernel/Read/ReadVariableOpReadVariableOp%transformer_encoder_23/dense_1/kernel*
_output_shapes

: *
dtype0
?
!transformer_encoder_23/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!transformer_encoder_23/dense/bias
?
5transformer_encoder_23/dense/bias/Read/ReadVariableOpReadVariableOp!transformer_encoder_23/dense/bias*
_output_shapes
: *
dtype0
?
#transformer_encoder_23/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *4
shared_name%#transformer_encoder_23/dense/kernel
?
7transformer_encoder_23/dense/kernel/Read/ReadVariableOpReadVariableOp#transformer_encoder_23/dense/kernel*
_output_shapes

: *
dtype0
?
1transformer_encoder_23/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31transformer_encoder_23/layer_normalization_1/beta
?
Etransformer_encoder_23/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp1transformer_encoder_23/layer_normalization_1/beta*
_output_shapes
:*
dtype0
?
2transformer_encoder_23/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42transformer_encoder_23/layer_normalization_1/gamma
?
Ftransformer_encoder_23/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp2transformer_encoder_23/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
?
/transformer_encoder_23/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_encoder_23/layer_normalization/beta
?
Ctransformer_encoder_23/layer_normalization/beta/Read/ReadVariableOpReadVariableOp/transformer_encoder_23/layer_normalization/beta*
_output_shapes
:*
dtype0
?
0transformer_encoder_23/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_encoder_23/layer_normalization/gamma
?
Dtransformer_encoder_23/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp0transformer_encoder_23/layer_normalization/gamma*
_output_shapes
:*
dtype0
?
Atransformer_encoder_23/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAtransformer_encoder_23/multi_head_attention/attention_output/bias
?
Utransformer_encoder_23/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpAtransformer_encoder_23/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
?
Ctransformer_encoder_23/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECtransformer_encoder_23/multi_head_attention/attention_output/kernel
?
Wtransformer_encoder_23/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpCtransformer_encoder_23/multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
?
6transformer_encoder_23/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_encoder_23/multi_head_attention/value/bias
?
Jtransformer_encoder_23/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp6transformer_encoder_23/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
?
8transformer_encoder_23/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_encoder_23/multi_head_attention/value/kernel
?
Ltransformer_encoder_23/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp8transformer_encoder_23/multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0
?
4transformer_encoder_23/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64transformer_encoder_23/multi_head_attention/key/bias
?
Htransformer_encoder_23/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp4transformer_encoder_23/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
?
6transformer_encoder_23/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86transformer_encoder_23/multi_head_attention/key/kernel
?
Jtransformer_encoder_23/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp6transformer_encoder_23/multi_head_attention/key/kernel*"
_output_shapes
:*
dtype0
?
6transformer_encoder_23/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_encoder_23/multi_head_attention/query/bias
?
Jtransformer_encoder_23/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp6transformer_encoder_23/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
?
8transformer_encoder_23/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8transformer_encoder_23/multi_head_attention/query/kernel
?
Ltransformer_encoder_23/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp8transformer_encoder_23/multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0
?
?token_and_position_embedding_28/position_embedding29/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *P
shared_nameA?token_and_position_embedding_28/position_embedding29/embeddings
?
Stoken_and_position_embedding_28/position_embedding29/embeddings/Read/ReadVariableOpReadVariableOp?token_and_position_embedding_28/position_embedding29/embeddings*
_output_shapes

: *
dtype0
?
<token_and_position_embedding_28/token_embedding29/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><token_and_position_embedding_28/token_embedding29/embeddings
?
Ptoken_and_position_embedding_28/token_embedding29/embeddings/Read/ReadVariableOpReadVariableOp<token_and_position_embedding_28/token_embedding29/embeddings*
_output_shapes

:*
dtype0
?
?token_and_position_embedding_27/position_embedding28/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *P
shared_nameA?token_and_position_embedding_27/position_embedding28/embeddings
?
Stoken_and_position_embedding_27/position_embedding28/embeddings/Read/ReadVariableOpReadVariableOp?token_and_position_embedding_27/position_embedding28/embeddings*
_output_shapes

: *
dtype0
?
<token_and_position_embedding_27/token_embedding28/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*M
shared_name><token_and_position_embedding_27/token_embedding28/embeddings
?
Ptoken_and_position_embedding_27/token_embedding28/embeddings/Read/ReadVariableOpReadVariableOp<token_and_position_embedding_27/token_embedding28/embeddings*
_output_shapes
:	?*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:@*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:@*
dtype0
y
serving_default_phrasePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_token_rolePlaceholder*'
_output_shapes
:????????? *
dtype0*
shape:????????? 
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_phraseserving_default_token_role
hash_tableConst_3Const_5Const_4<token_and_position_embedding_27/token_embedding28/embeddings?token_and_position_embedding_27/position_embedding28/embeddings<token_and_position_embedding_28/token_embedding29/embeddings?token_and_position_embedding_28/position_embedding29/embeddings8transformer_encoder_23/multi_head_attention/query/kernel6transformer_encoder_23/multi_head_attention/query/bias6transformer_encoder_23/multi_head_attention/key/kernel4transformer_encoder_23/multi_head_attention/key/bias8transformer_encoder_23/multi_head_attention/value/kernel6transformer_encoder_23/multi_head_attention/value/biasCtransformer_encoder_23/multi_head_attention/attention_output/kernelAtransformer_encoder_23/multi_head_attention/attention_output/bias0transformer_encoder_23/layer_normalization/gamma/transformer_encoder_23/layer_normalization/beta#transformer_encoder_23/dense/kernel!transformer_encoder_23/dense/bias%transformer_encoder_23/dense_1/kernel#transformer_encoder_23/dense_1/bias2transformer_encoder_23/layer_normalization_1/gamma1transformer_encoder_23/layer_normalization_1/beta8transformer_decoder_23/multi_head_attention/query/kernel6transformer_decoder_23/multi_head_attention/query/bias6transformer_decoder_23/multi_head_attention/key/kernel4transformer_decoder_23/multi_head_attention/key/bias8transformer_decoder_23/multi_head_attention/value/kernel6transformer_decoder_23/multi_head_attention/value/biasCtransformer_decoder_23/multi_head_attention/attention_output/kernelAtransformer_decoder_23/multi_head_attention/attention_output/bias0transformer_decoder_23/layer_normalization/gamma/transformer_decoder_23/layer_normalization/beta#transformer_decoder_23/dense/kernel!transformer_decoder_23/dense/bias%transformer_decoder_23/dense_1/kernel#transformer_decoder_23/dense_1/bias2transformer_decoder_23/layer_normalization_1/gamma1transformer_decoder_23/layer_normalization_1/betadense_23/kerneldense_23/biasdense_24/kerneldense_24/bias*9
Tin2
02.		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_763973
?
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__initializer_766044
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__initializer_766059
:
NoOpNoOp^PartitionedCall^StatefulPartitionedCall_1
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
??
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
;
	keras_api
_lookup_layer
_adapt_function*
* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
 position_embedding*
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'token_embedding
(position_embedding*
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_multi_head_attention_layer
6_attention_layernorm
7_feedforward_layernorm
8_attention_dropout
9_intermediate_dense
:_output_dense
;_output_dropout*
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_self_attention_layer
 C_decoder_attention_layernorm
D_feedforward_layernorm
E_self_attention_dropout
F_intermediate_dense
G_output_dense
H_output_dropout*
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias*
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator* 
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias*
?
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
U37
V38
d39
e40*
?
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
U36
V37
d38
e39*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateUm?Vm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Uv?Vv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 
* 
:
?	keras_api
?lookup_table
?token_counts*

?trace_0* 

f0
g1*

f0
g1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
f
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
g
embeddings
gposition_embeddings*

h0
i1*

h0
i1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
h
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
i
embeddings
iposition_embeddings*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
z
j0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12
w13
x14
y15*
z
j0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12
w13
x14
y15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	rgamma
sbeta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	tgamma
ubeta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

vkernel
wbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

xkernel
ybias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
z0
{1
|2
}3
~4
5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15*
?
z0
{1
|2
}3
~4
5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

U0
V1*

U0
V1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

d0
e1*

d0
e1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<token_and_position_embedding_27/token_embedding28/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?token_and_position_embedding_27/position_embedding28/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<token_and_position_embedding_28/token_embedding29/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?token_and_position_embedding_28/position_embedding29/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_encoder_23/multi_head_attention/query/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_encoder_23/multi_head_attention/query/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_encoder_23/multi_head_attention/key/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4transformer_encoder_23/multi_head_attention/key/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_encoder_23/multi_head_attention/value/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_encoder_23/multi_head_attention/value/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUECtransformer_encoder_23/multi_head_attention/attention_output/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAtransformer_encoder_23/multi_head_attention/attention_output/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_encoder_23/layer_normalization/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_encoder_23/layer_normalization/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2transformer_encoder_23/layer_normalization_1/gamma'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_encoder_23/layer_normalization_1/beta'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_encoder_23/dense/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_encoder_23/dense/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%transformer_encoder_23/dense_1/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_encoder_23/dense_1/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_decoder_23/multi_head_attention/query/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_decoder_23/multi_head_attention/query/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_decoder_23/multi_head_attention/key/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4transformer_decoder_23/multi_head_attention/key/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_decoder_23/multi_head_attention/value/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_decoder_23/multi_head_attention/value/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUECtransformer_decoder_23/multi_head_attention/attention_output/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAtransformer_decoder_23/multi_head_attention/attention_output/bias'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_decoder_23/layer_normalization/gamma'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_decoder_23/layer_normalization/beta'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2transformer_decoder_23/layer_normalization_1/gamma'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_decoder_23/layer_normalization_1/beta'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_decoder_23/dense/kernel'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_decoder_23/dense/bias'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%transformer_decoder_23/dense_1/kernel'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_decoder_23/dense_1/bias'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*

?0
?1*
* 
* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
2
?	capture_1
?	capture_2
?	capture_3* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
2
?	capture_1
?	capture_2
?	capture_3* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
?
?_create_resource
?_initialize
?_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*

?	capture_1* 
* 

0
 1*
* 
* 
* 
* 
* 

f0*

f0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

g0*

g0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

'0
(1*
* 
* 
* 
* 
* 

h0*

h0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

i0*

i0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
5
50
61
72
83
94
:5
;6*
* 
* 
* 
* 
* 
* 
* 
<
j0
k1
l2
m3
n4
o5
p6
q7*
<
j0
k1
l2
m3
n4
o5
p6
q7*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

jkernel
kbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

lkernel
mbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

nkernel
obias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

pkernel
qbias*

r0
s1*

r0
s1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

t0
u1*

t0
u1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

v0
w1*

v0
w1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

x0
y1*

x0
y1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
5
B0
C1
D2
E3
F4
G5
H6*
* 
* 
* 
* 
* 
* 
* 
>
z0
{1
|2
}3
~4
5
?6
?7*
>
z0
{1
|2
}3
~4
5
?6
?7*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

zkernel
{bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

|kernel
}bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

~kernel
bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
?0
?1
?2
?3
?4
?5*
* 
* 
* 

j0
k1*

j0
k1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

l0
m1*

l0
m1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

n0
o1*

n0
o1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

p0
q1*

p0
q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
?0
?1
?2
?3
?4
?5*
* 
* 
* 

z0
{1*

z0
{1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

|0
}1*

|0
}1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 

~0
1*

~0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
"
?	capture_1
?	capture_2* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_27/token_embedding28/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_27/position_embedding28/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_28/token_embedding29/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_28/position_embedding29/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_23/multi_head_attention/query/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_23/multi_head_attention/query/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_23/multi_head_attention/key/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_encoder_23/multi_head_attention/key/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_23/multi_head_attention/value/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_23/multi_head_attention/value/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_encoder_23/layer_normalization/gamma/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_encoder_23/layer_normalization/beta/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_encoder_23/layer_normalization_1/gamma/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_encoder_23/layer_normalization_1/beta/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_23/dense/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_encoder_23/dense/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_encoder_23/dense_1/kernel/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_23/dense_1/bias/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_23/multi_head_attention/query/kernel/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_23/multi_head_attention/query/bias/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_23/multi_head_attention/key/kernel/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_decoder_23/multi_head_attention/key/bias/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_23/multi_head_attention/value/kernel/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_23/multi_head_attention/value/bias/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_decoder_23/layer_normalization/gamma/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_decoder_23/layer_normalization/beta/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_decoder_23/layer_normalization_1/gamma/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_decoder_23/layer_normalization_1/beta/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_23/dense/kernel/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_decoder_23/dense/bias/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_decoder_23/dense_1/kernel/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_23/dense_1/bias/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_27/token_embedding28/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_27/position_embedding28/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_28/token_embedding29/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_28/position_embedding29/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_23/multi_head_attention/query/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_23/multi_head_attention/query/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_23/multi_head_attention/key/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_encoder_23/multi_head_attention/key/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_23/multi_head_attention/value/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_23/multi_head_attention/value/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_encoder_23/layer_normalization/gamma/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_encoder_23/layer_normalization/beta/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_encoder_23/layer_normalization_1/gamma/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_encoder_23/layer_normalization_1/beta/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_23/dense/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_encoder_23/dense/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_encoder_23/dense_1/kernel/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_23/dense_1/bias/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_23/multi_head_attention/query/kernel/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_23/multi_head_attention/query/bias/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_23/multi_head_attention/key/kernel/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_decoder_23/multi_head_attention/key/bias/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_23/multi_head_attention/value/kernel/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_23/multi_head_attention/value/bias/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_decoder_23/layer_normalization/gamma/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_decoder_23/layer_normalization/beta/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_decoder_23/layer_normalization_1/gamma/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_decoder_23/layer_normalization_1/beta/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_23/dense/kernel/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_decoder_23/dense/bias/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_decoder_23/dense_1/kernel/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_23/dense_1/bias/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?K
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpPtoken_and_position_embedding_27/token_embedding28/embeddings/Read/ReadVariableOpStoken_and_position_embedding_27/position_embedding28/embeddings/Read/ReadVariableOpPtoken_and_position_embedding_28/token_embedding29/embeddings/Read/ReadVariableOpStoken_and_position_embedding_28/position_embedding29/embeddings/Read/ReadVariableOpLtransformer_encoder_23/multi_head_attention/query/kernel/Read/ReadVariableOpJtransformer_encoder_23/multi_head_attention/query/bias/Read/ReadVariableOpJtransformer_encoder_23/multi_head_attention/key/kernel/Read/ReadVariableOpHtransformer_encoder_23/multi_head_attention/key/bias/Read/ReadVariableOpLtransformer_encoder_23/multi_head_attention/value/kernel/Read/ReadVariableOpJtransformer_encoder_23/multi_head_attention/value/bias/Read/ReadVariableOpWtransformer_encoder_23/multi_head_attention/attention_output/kernel/Read/ReadVariableOpUtransformer_encoder_23/multi_head_attention/attention_output/bias/Read/ReadVariableOpDtransformer_encoder_23/layer_normalization/gamma/Read/ReadVariableOpCtransformer_encoder_23/layer_normalization/beta/Read/ReadVariableOpFtransformer_encoder_23/layer_normalization_1/gamma/Read/ReadVariableOpEtransformer_encoder_23/layer_normalization_1/beta/Read/ReadVariableOp7transformer_encoder_23/dense/kernel/Read/ReadVariableOp5transformer_encoder_23/dense/bias/Read/ReadVariableOp9transformer_encoder_23/dense_1/kernel/Read/ReadVariableOp7transformer_encoder_23/dense_1/bias/Read/ReadVariableOpLtransformer_decoder_23/multi_head_attention/query/kernel/Read/ReadVariableOpJtransformer_decoder_23/multi_head_attention/query/bias/Read/ReadVariableOpJtransformer_decoder_23/multi_head_attention/key/kernel/Read/ReadVariableOpHtransformer_decoder_23/multi_head_attention/key/bias/Read/ReadVariableOpLtransformer_decoder_23/multi_head_attention/value/kernel/Read/ReadVariableOpJtransformer_decoder_23/multi_head_attention/value/bias/Read/ReadVariableOpWtransformer_decoder_23/multi_head_attention/attention_output/kernel/Read/ReadVariableOpUtransformer_decoder_23/multi_head_attention/attention_output/bias/Read/ReadVariableOpDtransformer_decoder_23/layer_normalization/gamma/Read/ReadVariableOpCtransformer_decoder_23/layer_normalization/beta/Read/ReadVariableOpFtransformer_decoder_23/layer_normalization_1/gamma/Read/ReadVariableOpEtransformer_decoder_23/layer_normalization_1/beta/Read/ReadVariableOp7transformer_decoder_23/dense/kernel/Read/ReadVariableOp5transformer_decoder_23/dense/bias/Read/ReadVariableOp9transformer_decoder_23/dense_1/kernel/Read/ReadVariableOp7transformer_decoder_23/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOpWAdam/token_and_position_embedding_27/token_embedding28/embeddings/m/Read/ReadVariableOpZAdam/token_and_position_embedding_27/position_embedding28/embeddings/m/Read/ReadVariableOpWAdam/token_and_position_embedding_28/token_embedding29/embeddings/m/Read/ReadVariableOpZAdam/token_and_position_embedding_28/position_embedding29/embeddings/m/Read/ReadVariableOpSAdam/transformer_encoder_23/multi_head_attention/query/kernel/m/Read/ReadVariableOpQAdam/transformer_encoder_23/multi_head_attention/query/bias/m/Read/ReadVariableOpQAdam/transformer_encoder_23/multi_head_attention/key/kernel/m/Read/ReadVariableOpOAdam/transformer_encoder_23/multi_head_attention/key/bias/m/Read/ReadVariableOpSAdam/transformer_encoder_23/multi_head_attention/value/kernel/m/Read/ReadVariableOpQAdam/transformer_encoder_23/multi_head_attention/value/bias/m/Read/ReadVariableOp^Adam/transformer_encoder_23/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOp\Adam/transformer_encoder_23/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpKAdam/transformer_encoder_23/layer_normalization/gamma/m/Read/ReadVariableOpJAdam/transformer_encoder_23/layer_normalization/beta/m/Read/ReadVariableOpMAdam/transformer_encoder_23/layer_normalization_1/gamma/m/Read/ReadVariableOpLAdam/transformer_encoder_23/layer_normalization_1/beta/m/Read/ReadVariableOp>Adam/transformer_encoder_23/dense/kernel/m/Read/ReadVariableOp<Adam/transformer_encoder_23/dense/bias/m/Read/ReadVariableOp@Adam/transformer_encoder_23/dense_1/kernel/m/Read/ReadVariableOp>Adam/transformer_encoder_23/dense_1/bias/m/Read/ReadVariableOpSAdam/transformer_decoder_23/multi_head_attention/query/kernel/m/Read/ReadVariableOpQAdam/transformer_decoder_23/multi_head_attention/query/bias/m/Read/ReadVariableOpQAdam/transformer_decoder_23/multi_head_attention/key/kernel/m/Read/ReadVariableOpOAdam/transformer_decoder_23/multi_head_attention/key/bias/m/Read/ReadVariableOpSAdam/transformer_decoder_23/multi_head_attention/value/kernel/m/Read/ReadVariableOpQAdam/transformer_decoder_23/multi_head_attention/value/bias/m/Read/ReadVariableOp^Adam/transformer_decoder_23/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOp\Adam/transformer_decoder_23/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpKAdam/transformer_decoder_23/layer_normalization/gamma/m/Read/ReadVariableOpJAdam/transformer_decoder_23/layer_normalization/beta/m/Read/ReadVariableOpMAdam/transformer_decoder_23/layer_normalization_1/gamma/m/Read/ReadVariableOpLAdam/transformer_decoder_23/layer_normalization_1/beta/m/Read/ReadVariableOp>Adam/transformer_decoder_23/dense/kernel/m/Read/ReadVariableOp<Adam/transformer_decoder_23/dense/bias/m/Read/ReadVariableOp@Adam/transformer_decoder_23/dense_1/kernel/m/Read/ReadVariableOp>Adam/transformer_decoder_23/dense_1/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOpWAdam/token_and_position_embedding_27/token_embedding28/embeddings/v/Read/ReadVariableOpZAdam/token_and_position_embedding_27/position_embedding28/embeddings/v/Read/ReadVariableOpWAdam/token_and_position_embedding_28/token_embedding29/embeddings/v/Read/ReadVariableOpZAdam/token_and_position_embedding_28/position_embedding29/embeddings/v/Read/ReadVariableOpSAdam/transformer_encoder_23/multi_head_attention/query/kernel/v/Read/ReadVariableOpQAdam/transformer_encoder_23/multi_head_attention/query/bias/v/Read/ReadVariableOpQAdam/transformer_encoder_23/multi_head_attention/key/kernel/v/Read/ReadVariableOpOAdam/transformer_encoder_23/multi_head_attention/key/bias/v/Read/ReadVariableOpSAdam/transformer_encoder_23/multi_head_attention/value/kernel/v/Read/ReadVariableOpQAdam/transformer_encoder_23/multi_head_attention/value/bias/v/Read/ReadVariableOp^Adam/transformer_encoder_23/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOp\Adam/transformer_encoder_23/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpKAdam/transformer_encoder_23/layer_normalization/gamma/v/Read/ReadVariableOpJAdam/transformer_encoder_23/layer_normalization/beta/v/Read/ReadVariableOpMAdam/transformer_encoder_23/layer_normalization_1/gamma/v/Read/ReadVariableOpLAdam/transformer_encoder_23/layer_normalization_1/beta/v/Read/ReadVariableOp>Adam/transformer_encoder_23/dense/kernel/v/Read/ReadVariableOp<Adam/transformer_encoder_23/dense/bias/v/Read/ReadVariableOp@Adam/transformer_encoder_23/dense_1/kernel/v/Read/ReadVariableOp>Adam/transformer_encoder_23/dense_1/bias/v/Read/ReadVariableOpSAdam/transformer_decoder_23/multi_head_attention/query/kernel/v/Read/ReadVariableOpQAdam/transformer_decoder_23/multi_head_attention/query/bias/v/Read/ReadVariableOpQAdam/transformer_decoder_23/multi_head_attention/key/kernel/v/Read/ReadVariableOpOAdam/transformer_decoder_23/multi_head_attention/key/bias/v/Read/ReadVariableOpSAdam/transformer_decoder_23/multi_head_attention/value/kernel/v/Read/ReadVariableOpQAdam/transformer_decoder_23/multi_head_attention/value/bias/v/Read/ReadVariableOp^Adam/transformer_decoder_23/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOp\Adam/transformer_decoder_23/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpKAdam/transformer_decoder_23/layer_normalization/gamma/v/Read/ReadVariableOpJAdam/transformer_decoder_23/layer_normalization/beta/v/Read/ReadVariableOpMAdam/transformer_decoder_23/layer_normalization_1/gamma/v/Read/ReadVariableOpLAdam/transformer_decoder_23/layer_normalization_1/beta/v/Read/ReadVariableOp>Adam/transformer_decoder_23/dense/kernel/v/Read/ReadVariableOp<Adam/transformer_decoder_23/dense/bias/v/Read/ReadVariableOp@Adam/transformer_decoder_23/dense_1/kernel/v/Read/ReadVariableOp>Adam/transformer_decoder_23/dense_1/bias/v/Read/ReadVariableOpConst_6*?
Tin?
?2?		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_766517
?6
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/biasdense_24/kerneldense_24/bias<token_and_position_embedding_27/token_embedding28/embeddings?token_and_position_embedding_27/position_embedding28/embeddings<token_and_position_embedding_28/token_embedding29/embeddings?token_and_position_embedding_28/position_embedding29/embeddings8transformer_encoder_23/multi_head_attention/query/kernel6transformer_encoder_23/multi_head_attention/query/bias6transformer_encoder_23/multi_head_attention/key/kernel4transformer_encoder_23/multi_head_attention/key/bias8transformer_encoder_23/multi_head_attention/value/kernel6transformer_encoder_23/multi_head_attention/value/biasCtransformer_encoder_23/multi_head_attention/attention_output/kernelAtransformer_encoder_23/multi_head_attention/attention_output/bias0transformer_encoder_23/layer_normalization/gamma/transformer_encoder_23/layer_normalization/beta2transformer_encoder_23/layer_normalization_1/gamma1transformer_encoder_23/layer_normalization_1/beta#transformer_encoder_23/dense/kernel!transformer_encoder_23/dense/bias%transformer_encoder_23/dense_1/kernel#transformer_encoder_23/dense_1/bias8transformer_decoder_23/multi_head_attention/query/kernel6transformer_decoder_23/multi_head_attention/query/bias6transformer_decoder_23/multi_head_attention/key/kernel4transformer_decoder_23/multi_head_attention/key/bias8transformer_decoder_23/multi_head_attention/value/kernel6transformer_decoder_23/multi_head_attention/value/biasCtransformer_decoder_23/multi_head_attention/attention_output/kernelAtransformer_decoder_23/multi_head_attention/attention_output/bias0transformer_decoder_23/layer_normalization/gamma/transformer_decoder_23/layer_normalization/beta2transformer_decoder_23/layer_normalization_1/gamma1transformer_decoder_23/layer_normalization_1/beta#transformer_decoder_23/dense/kernel!transformer_decoder_23/dense/bias%transformer_decoder_23/dense_1/kernel#transformer_decoder_23/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mCAdam/token_and_position_embedding_27/token_embedding28/embeddings/mFAdam/token_and_position_embedding_27/position_embedding28/embeddings/mCAdam/token_and_position_embedding_28/token_embedding29/embeddings/mFAdam/token_and_position_embedding_28/position_embedding29/embeddings/m?Adam/transformer_encoder_23/multi_head_attention/query/kernel/m=Adam/transformer_encoder_23/multi_head_attention/query/bias/m=Adam/transformer_encoder_23/multi_head_attention/key/kernel/m;Adam/transformer_encoder_23/multi_head_attention/key/bias/m?Adam/transformer_encoder_23/multi_head_attention/value/kernel/m=Adam/transformer_encoder_23/multi_head_attention/value/bias/mJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/mHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/m7Adam/transformer_encoder_23/layer_normalization/gamma/m6Adam/transformer_encoder_23/layer_normalization/beta/m9Adam/transformer_encoder_23/layer_normalization_1/gamma/m8Adam/transformer_encoder_23/layer_normalization_1/beta/m*Adam/transformer_encoder_23/dense/kernel/m(Adam/transformer_encoder_23/dense/bias/m,Adam/transformer_encoder_23/dense_1/kernel/m*Adam/transformer_encoder_23/dense_1/bias/m?Adam/transformer_decoder_23/multi_head_attention/query/kernel/m=Adam/transformer_decoder_23/multi_head_attention/query/bias/m=Adam/transformer_decoder_23/multi_head_attention/key/kernel/m;Adam/transformer_decoder_23/multi_head_attention/key/bias/m?Adam/transformer_decoder_23/multi_head_attention/value/kernel/m=Adam/transformer_decoder_23/multi_head_attention/value/bias/mJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/mHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/m7Adam/transformer_decoder_23/layer_normalization/gamma/m6Adam/transformer_decoder_23/layer_normalization/beta/m9Adam/transformer_decoder_23/layer_normalization_1/gamma/m8Adam/transformer_decoder_23/layer_normalization_1/beta/m*Adam/transformer_decoder_23/dense/kernel/m(Adam/transformer_decoder_23/dense/bias/m,Adam/transformer_decoder_23/dense_1/kernel/m*Adam/transformer_decoder_23/dense_1/bias/mAdam/dense_23/kernel/vAdam/dense_23/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vCAdam/token_and_position_embedding_27/token_embedding28/embeddings/vFAdam/token_and_position_embedding_27/position_embedding28/embeddings/vCAdam/token_and_position_embedding_28/token_embedding29/embeddings/vFAdam/token_and_position_embedding_28/position_embedding29/embeddings/v?Adam/transformer_encoder_23/multi_head_attention/query/kernel/v=Adam/transformer_encoder_23/multi_head_attention/query/bias/v=Adam/transformer_encoder_23/multi_head_attention/key/kernel/v;Adam/transformer_encoder_23/multi_head_attention/key/bias/v?Adam/transformer_encoder_23/multi_head_attention/value/kernel/v=Adam/transformer_encoder_23/multi_head_attention/value/bias/vJAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/vHAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/v7Adam/transformer_encoder_23/layer_normalization/gamma/v6Adam/transformer_encoder_23/layer_normalization/beta/v9Adam/transformer_encoder_23/layer_normalization_1/gamma/v8Adam/transformer_encoder_23/layer_normalization_1/beta/v*Adam/transformer_encoder_23/dense/kernel/v(Adam/transformer_encoder_23/dense/bias/v,Adam/transformer_encoder_23/dense_1/kernel/v*Adam/transformer_encoder_23/dense_1/bias/v?Adam/transformer_decoder_23/multi_head_attention/query/kernel/v=Adam/transformer_decoder_23/multi_head_attention/query/bias/v=Adam/transformer_decoder_23/multi_head_attention/key/kernel/v;Adam/transformer_decoder_23/multi_head_attention/key/bias/v?Adam/transformer_decoder_23/multi_head_attention/value/kernel/v=Adam/transformer_decoder_23/multi_head_attention/value/bias/vJAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/vHAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/v7Adam/transformer_decoder_23/layer_normalization/gamma/v6Adam/transformer_decoder_23/layer_normalization/beta/v9Adam/transformer_decoder_23/layer_normalization_1/gamma/v8Adam/transformer_decoder_23/layer_normalization_1/beta/v*Adam/transformer_decoder_23/dense/kernel/v(Adam/transformer_decoder_23/dense/bias/v,Adam/transformer_decoder_23/dense_1/kernel/v*Adam/transformer_decoder_23/dense_1/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_766926??.
?"
?
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_762085

inputs;
)token_embedding29_embedding_lookup_762061:>
,position_embedding29_readvariableop_resource: 
identity??#position_embedding29/ReadVariableOp?"token_embedding29/embedding_lookupg
token_embedding29/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
"token_embedding29/embedding_lookupResourceGather)token_embedding29_embedding_lookup_762061token_embedding29/Cast:y:0*
Tindices0*<
_class2
0.loc:@token_embedding29/embedding_lookup/762061*+
_output_shapes
:????????? *
dtype0?
+token_embedding29/embedding_lookup/IdentityIdentity+token_embedding29/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding29/embedding_lookup/762061*+
_output_shapes
:????????? ?
-token_embedding29/embedding_lookup/Identity_1Identity4token_embedding29/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
position_embedding29/ShapeShape6token_embedding29/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding29/strided_sliceStridedSlice#position_embedding29/Shape:output:01position_embedding29/strided_slice/stack:output:03position_embedding29/strided_slice/stack_1:output:03position_embedding29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding29/ReadVariableOpReadVariableOp,position_embedding29_readvariableop_resource*
_output_shapes

: *
dtype0\
position_embedding29/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding29/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding29/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding29/strided_slice_1/stackPack#position_embedding29/Const:output:05position_embedding29/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding29/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding29/strided_slice_1/stack_1Pack+position_embedding29/strided_slice:output:07position_embedding29/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding29/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding29/strided_slice_1/stack_2Pack%position_embedding29/Const_1:output:07position_embedding29/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding29/strided_slice_1StridedSlice+position_embedding29/ReadVariableOp:value:03position_embedding29/strided_slice_1/stack:output:05position_embedding29/strided_slice_1/stack_1:output:05position_embedding29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
 position_embedding29/BroadcastToBroadcastTo-position_embedding29/strided_slice_1:output:0#position_embedding29/Shape:output:0*
T0*+
_output_shapes
:????????? ?
addAddV26token_embedding29/embedding_lookup/Identity_1:output:0)position_embedding29/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp$^position_embedding29/ReadVariableOp#^token_embedding29/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding29/ReadVariableOp#position_embedding29/ReadVariableOp2H
"token_embedding29/embedding_lookup"token_embedding29/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
՜
?
D__inference_model_12_layer_call_and_return_conditional_losses_763402

inputs
inputs_1U
Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_13_string_lookup_13_equal_y5
1text_vectorization_13_string_lookup_13_selectv2_t	9
&token_and_position_embedding_27_763312:	?8
&token_and_position_embedding_27_763314: 8
&token_and_position_embedding_28_763317:8
&token_and_position_embedding_28_763319: 3
transformer_encoder_23_763323:/
transformer_encoder_23_763325:3
transformer_encoder_23_763327:/
transformer_encoder_23_763329:3
transformer_encoder_23_763331:/
transformer_encoder_23_763333:3
transformer_encoder_23_763335:+
transformer_encoder_23_763337:+
transformer_encoder_23_763339:+
transformer_encoder_23_763341:/
transformer_encoder_23_763343: +
transformer_encoder_23_763345: /
transformer_encoder_23_763347: +
transformer_encoder_23_763349:+
transformer_encoder_23_763351:+
transformer_encoder_23_763353:3
transformer_decoder_23_763356:/
transformer_decoder_23_763358:3
transformer_decoder_23_763360:/
transformer_decoder_23_763362:3
transformer_decoder_23_763364:/
transformer_decoder_23_763366:3
transformer_decoder_23_763368:+
transformer_decoder_23_763370:+
transformer_decoder_23_763372:+
transformer_decoder_23_763374:/
transformer_decoder_23_763376: +
transformer_decoder_23_763378: /
transformer_decoder_23_763380: +
transformer_decoder_23_763382:+
transformer_decoder_23_763384:+
transformer_decoder_23_763386:!
dense_23_763390:@
dense_23_763392:@!
dense_24_763396:@
dense_24_763398:
identity?? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?7token_and_position_embedding_27/StatefulPartitionedCall?7token_and_position_embedding_28/StatefulPartitionedCall?.transformer_decoder_23/StatefulPartitionedCall?.transformer_encoder_23/StatefulPartitionedCall~
text_vectorization_13/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????h
'text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2&text_vectorization_13/Squeeze:output:00text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
5text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/text_vectorization_13/StringSplit/strided_sliceStridedSlice9text_vectorization_13/StringSplit/StringSplitV2:indices:0>text_vectorization_13/StringSplit/strided_slice/stack:output:0@text_vectorization_13/StringSplit/strided_slice/stack_1:output:0@text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
7text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1text_vectorization_13/StringSplit/strided_slice_1StridedSlice7text_vectorization_13/StringSplit/StringSplitV2:shape:0@text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Xtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ftext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0stext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
etext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountmtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ctext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle8text_vectorization_13/StringSplit/StringSplitV2:values:0Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_13/string_lookup_13/EqualEqual8text_vectorization_13/StringSplit/StringSplitV2:values:0.text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/SelectV2SelectV20text_vectorization_13/string_lookup_13/Equal:z:01text_vectorization_13_string_lookup_13_selectv2_tMtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/IdentityIdentity8text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????t
2text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
*text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
9text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_13/RaggedToTensor/Const:output:08text_vectorization_13/string_lookup_13/Identity:output:0;text_vectorization_13/RaggedToTensor/default_value:output:0:text_vectorization_13/StringSplit/strided_slice_1:output:08text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_27/StatefulPartitionedCallStatefulPartitionedCallBtext_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_27_763312&token_and_position_embedding_27_763314*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_762051?
7token_and_position_embedding_28/StatefulPartitionedCallStatefulPartitionedCallinputs_1&token_and_position_embedding_28_763317&token_and_position_embedding_28_763319*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_762085?
add_12/PartitionedCallPartitionedCall@token_and_position_embedding_27/StatefulPartitionedCall:output:0@token_and_position_embedding_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_762097?
.transformer_encoder_23/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0transformer_encoder_23_763323transformer_encoder_23_763325transformer_encoder_23_763327transformer_encoder_23_763329transformer_encoder_23_763331transformer_encoder_23_763333transformer_encoder_23_763335transformer_encoder_23_763337transformer_encoder_23_763339transformer_encoder_23_763341transformer_encoder_23_763343transformer_encoder_23_763345transformer_encoder_23_763347transformer_encoder_23_763349transformer_encoder_23_763351transformer_encoder_23_763353*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_763101?
.transformer_decoder_23/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_23/StatefulPartitionedCall:output:0transformer_decoder_23_763356transformer_decoder_23_763358transformer_decoder_23_763360transformer_decoder_23_763362transformer_decoder_23_763364transformer_decoder_23_763366transformer_decoder_23_763368transformer_decoder_23_763370transformer_decoder_23_763372transformer_decoder_23_763374transformer_decoder_23_763376transformer_decoder_23_763378transformer_decoder_23_763380transformer_decoder_23_763382transformer_decoder_23_763384transformer_decoder_23_763386*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762878?
+global_average_pooling1d_12/PartitionedCallPartitionedCall7transformer_decoder_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_761964?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0dense_23_763390dense_23_763392*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_762479?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_762631?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_24_763396dense_24_763398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_762503x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCallE^text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV28^token_and_position_embedding_27/StatefulPartitionedCall8^token_and_position_embedding_28/StatefulPartitionedCall/^transformer_decoder_23/StatefulPartitionedCall/^transformer_encoder_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2?
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_27/StatefulPartitionedCall7token_and_position_embedding_27/StatefulPartitionedCall2r
7token_and_position_embedding_28/StatefulPartitionedCall7token_and_position_embedding_28/StatefulPartitionedCall2`
.transformer_decoder_23/StatefulPartitionedCall.transformer_decoder_23/StatefulPartitionedCall2`
.transformer_encoder_23/StatefulPartitionedCall.transformer_encoder_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?

$__inference_signature_wrapper_763973

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:

unknown_37:

unknown_38:

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallphrase
token_roleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_761954o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namephrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
token_role:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_save_fn_766083
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
;
__inference__creator_766036
identity??
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name614410*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765512

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  q
,multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*multi_head_attention/dropout_2/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:05multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Cmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0z
5multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
+multi_head_attention/dropout_2/dropout/CastCast7multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/Mul_1Mul.multi_head_attention/dropout_2/dropout/Mul:z:0/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? e
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_transformer_encoder_23_layer_call_fn_765200

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_762226s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
G
+__inference_dropout_10_layer_call_fn_765989

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_762490`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_dense_24_layer_call_and_return_conditional_losses_766031

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?;
!__inference__wrapped_model_761954

phrase

token_role^
Zmodel_12_text_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle_
[model_12_text_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	;
7model_12_text_vectorization_13_string_lookup_13_equal_y>
:model_12_text_vectorization_13_string_lookup_13_selectv2_t	e
Rmodel_12_token_and_position_embedding_27_token_embedding28_embedding_lookup_761596:	?g
Umodel_12_token_and_position_embedding_27_position_embedding28_readvariableop_resource: d
Rmodel_12_token_and_position_embedding_28_token_embedding29_embedding_lookup_761620:g
Umodel_12_token_and_position_embedding_28_position_embedding29_readvariableop_resource: v
`model_12_transformer_encoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource:h
Vmodel_12_transformer_encoder_23_multi_head_attention_query_add_readvariableop_resource:t
^model_12_transformer_encoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource:f
Tmodel_12_transformer_encoder_23_multi_head_attention_key_add_readvariableop_resource:v
`model_12_transformer_encoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource:h
Vmodel_12_transformer_encoder_23_multi_head_attention_value_add_readvariableop_resource:?
kmodel_12_transformer_encoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:o
amodel_12_transformer_encoder_23_multi_head_attention_attention_output_add_readvariableop_resource:g
Ymodel_12_transformer_encoder_23_layer_normalization_batchnorm_mul_readvariableop_resource:c
Umodel_12_transformer_encoder_23_layer_normalization_batchnorm_readvariableop_resource:Y
Gmodel_12_transformer_encoder_23_dense_tensordot_readvariableop_resource: S
Emodel_12_transformer_encoder_23_dense_biasadd_readvariableop_resource: [
Imodel_12_transformer_encoder_23_dense_1_tensordot_readvariableop_resource: U
Gmodel_12_transformer_encoder_23_dense_1_biasadd_readvariableop_resource:i
[model_12_transformer_encoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource:e
Wmodel_12_transformer_encoder_23_layer_normalization_1_batchnorm_readvariableop_resource:v
`model_12_transformer_decoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource:h
Vmodel_12_transformer_decoder_23_multi_head_attention_query_add_readvariableop_resource:t
^model_12_transformer_decoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource:f
Tmodel_12_transformer_decoder_23_multi_head_attention_key_add_readvariableop_resource:v
`model_12_transformer_decoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource:h
Vmodel_12_transformer_decoder_23_multi_head_attention_value_add_readvariableop_resource:?
kmodel_12_transformer_decoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:o
amodel_12_transformer_decoder_23_multi_head_attention_attention_output_add_readvariableop_resource:g
Ymodel_12_transformer_decoder_23_layer_normalization_batchnorm_mul_readvariableop_resource:c
Umodel_12_transformer_decoder_23_layer_normalization_batchnorm_readvariableop_resource:Y
Gmodel_12_transformer_decoder_23_dense_tensordot_readvariableop_resource: S
Emodel_12_transformer_decoder_23_dense_biasadd_readvariableop_resource: [
Imodel_12_transformer_decoder_23_dense_1_tensordot_readvariableop_resource: U
Gmodel_12_transformer_decoder_23_dense_1_biasadd_readvariableop_resource:i
[model_12_transformer_decoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource:e
Wmodel_12_transformer_decoder_23_layer_normalization_1_batchnorm_readvariableop_resource:B
0model_12_dense_23_matmul_readvariableop_resource:@?
1model_12_dense_23_biasadd_readvariableop_resource:@B
0model_12_dense_24_matmul_readvariableop_resource:@?
1model_12_dense_24_biasadd_readvariableop_resource:
identity??(model_12/dense_23/BiasAdd/ReadVariableOp?'model_12/dense_23/MatMul/ReadVariableOp?(model_12/dense_24/BiasAdd/ReadVariableOp?'model_12/dense_24/MatMul/ReadVariableOp?Mmodel_12/text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?Lmodel_12/token_and_position_embedding_27/position_embedding28/ReadVariableOp?Kmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookup?Lmodel_12/token_and_position_embedding_28/position_embedding29/ReadVariableOp?Kmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookup?<model_12/transformer_decoder_23/dense/BiasAdd/ReadVariableOp?>model_12/transformer_decoder_23/dense/Tensordot/ReadVariableOp?>model_12/transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp?@model_12/transformer_decoder_23/dense_1/Tensordot/ReadVariableOp?Lmodel_12/transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp?Pmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp?Nmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp?Rmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp?Xmodel_12/transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp?bmodel_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Kmodel_12/transformer_decoder_23/multi_head_attention/key/add/ReadVariableOp?Umodel_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Mmodel_12/transformer_decoder_23/multi_head_attention/query/add/ReadVariableOp?Wmodel_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Mmodel_12/transformer_decoder_23/multi_head_attention/value/add/ReadVariableOp?Wmodel_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp?<model_12/transformer_encoder_23/dense/BiasAdd/ReadVariableOp?>model_12/transformer_encoder_23/dense/Tensordot/ReadVariableOp?>model_12/transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp?@model_12/transformer_encoder_23/dense_1/Tensordot/ReadVariableOp?Lmodel_12/transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp?Pmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp?Nmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp?Rmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp?Xmodel_12/transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp?bmodel_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Kmodel_12/transformer_encoder_23/multi_head_attention/key/add/ReadVariableOp?Umodel_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Mmodel_12/transformer_encoder_23/multi_head_attention/query/add/ReadVariableOp?Wmodel_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Mmodel_12/transformer_encoder_23/multi_head_attention/value/add/ReadVariableOp?Wmodel_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
&model_12/text_vectorization_13/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????q
0model_12/text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
8model_12/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2/model_12/text_vectorization_13/Squeeze:output:09model_12/text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
>model_12/text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
@model_12/text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
@model_12/text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
8model_12/text_vectorization_13/StringSplit/strided_sliceStridedSliceBmodel_12/text_vectorization_13/StringSplit/StringSplitV2:indices:0Gmodel_12/text_vectorization_13/StringSplit/strided_slice/stack:output:0Imodel_12/text_vectorization_13/StringSplit/strided_slice/stack_1:output:0Imodel_12/text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
@model_12/text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Bmodel_12/text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_12/text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
:model_12/text_vectorization_13/StringSplit/strided_slice_1StridedSlice@model_12/text_vectorization_13/StringSplit/StringSplitV2:shape:0Imodel_12/text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Kmodel_12/text_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Kmodel_12/text_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
amodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastAmodel_12/text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
cmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastCmodel_12/text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
kmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeemodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
kmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
jmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdtmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0tmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
omodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatersmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0xmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
jmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastqmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
imodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxemodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0vmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
kmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
imodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2rmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0tmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
imodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulnmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumgmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumgmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0qmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
smodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
mmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeemodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0|model_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
nmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountvmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0qmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0vmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
hmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
cmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumumodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0qmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
lmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
hmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
cmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2umodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0imodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0qmodel_12/text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Mmodel_12/text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Zmodel_12_text_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleAmodel_12/text_vectorization_13/StringSplit/StringSplitV2:values:0[model_12_text_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
5model_12/text_vectorization_13/string_lookup_13/EqualEqualAmodel_12/text_vectorization_13/StringSplit/StringSplitV2:values:07model_12_text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
8model_12/text_vectorization_13/string_lookup_13/SelectV2SelectV29model_12/text_vectorization_13/string_lookup_13/Equal:z:0:model_12_text_vectorization_13_string_lookup_13_selectv2_tVmodel_12/text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
8model_12/text_vectorization_13/string_lookup_13/IdentityIdentityAmodel_12/text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????}
;model_12/text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
3model_12/text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
Bmodel_12/text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor<model_12/text_vectorization_13/RaggedToTensor/Const:output:0Amodel_12/text_vectorization_13/string_lookup_13/Identity:output:0Dmodel_12/text_vectorization_13/RaggedToTensor/default_value:output:0Cmodel_12/text_vectorization_13/StringSplit/strided_slice_1:output:0Amodel_12/text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Kmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookupResourceGatherRmodel_12_token_and_position_embedding_27_token_embedding28_embedding_lookup_761596Kmodel_12/text_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*e
_class[
YWloc:@model_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/761596*+
_output_shapes
:????????? *
dtype0?
Tmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/IdentityIdentityTmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookup:output:0*
T0*e
_class[
YWloc:@model_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/761596*+
_output_shapes
:????????? ?
Vmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1Identity]model_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/token_and_position_embedding_27/position_embedding28/ShapeShape_model_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Qmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Smodel_12/token_and_position_embedding_27/position_embedding28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Smodel_12/token_and_position_embedding_27/position_embedding28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Kmodel_12/token_and_position_embedding_27/position_embedding28/strided_sliceStridedSliceLmodel_12/token_and_position_embedding_27/position_embedding28/Shape:output:0Zmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice/stack:output:0\model_12/token_and_position_embedding_27/position_embedding28/strided_slice/stack_1:output:0\model_12/token_and_position_embedding_27/position_embedding28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Lmodel_12/token_and_position_embedding_27/position_embedding28/ReadVariableOpReadVariableOpUmodel_12_token_and_position_embedding_27_position_embedding28_readvariableop_resource*
_output_shapes

: *
dtype0?
Cmodel_12/token_and_position_embedding_27/position_embedding28/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Emodel_12/token_and_position_embedding_27/position_embedding28/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Umodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Smodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stackPackLmodel_12/token_and_position_embedding_27/position_embedding28/Const:output:0^model_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Wmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Umodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1PackTmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice:output:0`model_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Wmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Umodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2PackNmodel_12/token_and_position_embedding_27/position_embedding28/Const_1:output:0`model_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Mmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1StridedSliceTmodel_12/token_and_position_embedding_27/position_embedding28/ReadVariableOp:value:0\model_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack:output:0^model_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1:output:0^model_12/token_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
Imodel_12/token_and_position_embedding_27/position_embedding28/BroadcastToBroadcastToVmodel_12/token_and_position_embedding_27/position_embedding28/strided_slice_1:output:0Lmodel_12/token_and_position_embedding_27/position_embedding28/Shape:output:0*
T0*+
_output_shapes
:????????? ?
,model_12/token_and_position_embedding_27/addAddV2_model_12/token_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1:output:0Rmodel_12/token_and_position_embedding_27/position_embedding28/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? ?
?model_12/token_and_position_embedding_28/token_embedding29/CastCast
token_role*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
Kmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookupResourceGatherRmodel_12_token_and_position_embedding_28_token_embedding29_embedding_lookup_761620Cmodel_12/token_and_position_embedding_28/token_embedding29/Cast:y:0*
Tindices0*e
_class[
YWloc:@model_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/761620*+
_output_shapes
:????????? *
dtype0?
Tmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/IdentityIdentityTmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookup:output:0*
T0*e
_class[
YWloc:@model_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/761620*+
_output_shapes
:????????? ?
Vmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1Identity]model_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/token_and_position_embedding_28/position_embedding29/ShapeShape_model_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Qmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Smodel_12/token_and_position_embedding_28/position_embedding29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Smodel_12/token_and_position_embedding_28/position_embedding29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Kmodel_12/token_and_position_embedding_28/position_embedding29/strided_sliceStridedSliceLmodel_12/token_and_position_embedding_28/position_embedding29/Shape:output:0Zmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice/stack:output:0\model_12/token_and_position_embedding_28/position_embedding29/strided_slice/stack_1:output:0\model_12/token_and_position_embedding_28/position_embedding29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Lmodel_12/token_and_position_embedding_28/position_embedding29/ReadVariableOpReadVariableOpUmodel_12_token_and_position_embedding_28_position_embedding29_readvariableop_resource*
_output_shapes

: *
dtype0?
Cmodel_12/token_and_position_embedding_28/position_embedding29/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Emodel_12/token_and_position_embedding_28/position_embedding29/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Umodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Smodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stackPackLmodel_12/token_and_position_embedding_28/position_embedding29/Const:output:0^model_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Wmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Umodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1PackTmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice:output:0`model_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Wmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Umodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2PackNmodel_12/token_and_position_embedding_28/position_embedding29/Const_1:output:0`model_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Mmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1StridedSliceTmodel_12/token_and_position_embedding_28/position_embedding29/ReadVariableOp:value:0\model_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack:output:0^model_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1:output:0^model_12/token_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
Imodel_12/token_and_position_embedding_28/position_embedding29/BroadcastToBroadcastToVmodel_12/token_and_position_embedding_28/position_embedding29/strided_slice_1:output:0Lmodel_12/token_and_position_embedding_28/position_embedding29/Shape:output:0*
T0*+
_output_shapes
:????????? ?
,model_12/token_and_position_embedding_28/addAddV2_model_12/token_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1:output:0Rmodel_12/token_and_position_embedding_28/position_embedding29/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? ?
model_12/add_12/addAddV20model_12/token_and_position_embedding_27/add:z:00model_12/token_and_position_embedding_28/add:z:0*
T0*+
_output_shapes
:????????? ?
Wmodel_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp`model_12_transformer_encoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Hmodel_12/transformer_encoder_23/multi_head_attention/query/einsum/EinsumEinsummodel_12/add_12/add:z:0_model_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Mmodel_12/transformer_encoder_23/multi_head_attention/query/add/ReadVariableOpReadVariableOpVmodel_12_transformer_encoder_23_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
>model_12/transformer_encoder_23/multi_head_attention/query/addAddV2Qmodel_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum:output:0Umodel_12/transformer_encoder_23/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Umodel_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp^model_12_transformer_encoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fmodel_12/transformer_encoder_23/multi_head_attention/key/einsum/EinsumEinsummodel_12/add_12/add:z:0]model_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Kmodel_12/transformer_encoder_23/multi_head_attention/key/add/ReadVariableOpReadVariableOpTmodel_12_transformer_encoder_23_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
<model_12/transformer_encoder_23/multi_head_attention/key/addAddV2Omodel_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum:output:0Smodel_12/transformer_encoder_23/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Wmodel_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp`model_12_transformer_encoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Hmodel_12/transformer_encoder_23/multi_head_attention/value/einsum/EinsumEinsummodel_12/add_12/add:z:0_model_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Mmodel_12/transformer_encoder_23/multi_head_attention/value/add/ReadVariableOpReadVariableOpVmodel_12_transformer_encoder_23_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
>model_12/transformer_encoder_23/multi_head_attention/value/addAddV2Qmodel_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum:output:0Umodel_12/transformer_encoder_23/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
:model_12/transformer_encoder_23/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
8model_12/transformer_encoder_23/multi_head_attention/MulMulBmodel_12/transformer_encoder_23/multi_head_attention/query/add:z:0Cmodel_12/transformer_encoder_23/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
Bmodel_12/transformer_encoder_23/multi_head_attention/einsum/EinsumEinsum@model_12/transformer_encoder_23/multi_head_attention/key/add:z:0<model_12/transformer_encoder_23/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
Dmodel_12/transformer_encoder_23/multi_head_attention/softmax/SoftmaxSoftmaxKmodel_12/transformer_encoder_23/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
Gmodel_12/transformer_encoder_23/multi_head_attention/dropout_2/IdentityIdentityNmodel_12/transformer_encoder_23/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
Dmodel_12/transformer_encoder_23/multi_head_attention/einsum_1/EinsumEinsumPmodel_12/transformer_encoder_23/multi_head_attention/dropout_2/Identity:output:0Bmodel_12/transformer_encoder_23/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
bmodel_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpkmodel_12_transformer_encoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Smodel_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/EinsumEinsumMmodel_12/transformer_encoder_23/multi_head_attention/einsum_1/Einsum:output:0jmodel_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Xmodel_12/transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpamodel_12_transformer_encoder_23_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
Imodel_12/transformer_encoder_23/multi_head_attention/attention_output/addAddV2\model_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum:output:0`model_12/transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
0model_12/transformer_encoder_23/dropout/IdentityIdentityMmodel_12/transformer_encoder_23/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
#model_12/transformer_encoder_23/addAddV2model_12/add_12/add:z:09model_12/transformer_encoder_23/dropout/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Rmodel_12/transformer_encoder_23/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_12/transformer_encoder_23/layer_normalization/moments/meanMean'model_12/transformer_encoder_23/add:z:0[model_12/transformer_encoder_23/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Hmodel_12/transformer_encoder_23/layer_normalization/moments/StopGradientStopGradientImodel_12/transformer_encoder_23/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Mmodel_12/transformer_encoder_23/layer_normalization/moments/SquaredDifferenceSquaredDifference'model_12/transformer_encoder_23/add:z:0Qmodel_12/transformer_encoder_23/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Vmodel_12/transformer_encoder_23/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_12/transformer_encoder_23/layer_normalization/moments/varianceMeanQmodel_12/transformer_encoder_23/layer_normalization/moments/SquaredDifference:z:0_model_12/transformer_encoder_23/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Cmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Amodel_12/transformer_encoder_23/layer_normalization/batchnorm/addAddV2Mmodel_12/transformer_encoder_23/layer_normalization/moments/variance:output:0Lmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_encoder_23/layer_normalization/batchnorm/RsqrtRsqrtEmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Pmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_12_transformer_encoder_23_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_12/transformer_encoder_23/layer_normalization/batchnorm/mulMulGmodel_12/transformer_encoder_23/layer_normalization/batchnorm/Rsqrt:y:0Xmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul_1Mul'model_12/transformer_encoder_23/add:z:0Emodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul_2MulImodel_12/transformer_encoder_23/layer_normalization/moments/mean:output:0Emodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Lmodel_12/transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpReadVariableOpUmodel_12_transformer_encoder_23_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_12/transformer_encoder_23/layer_normalization/batchnorm/subSubTmodel_12/transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp:value:0Gmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add_1AddV2Gmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul_1:z:0Emodel_12/transformer_encoder_23/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
>model_12/transformer_encoder_23/dense/Tensordot/ReadVariableOpReadVariableOpGmodel_12_transformer_encoder_23_dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0~
4model_12/transformer_encoder_23/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
4model_12/transformer_encoder_23/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
5model_12/transformer_encoder_23/dense/Tensordot/ShapeShapeGmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
=model_12/transformer_encoder_23/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_12/transformer_encoder_23/dense/Tensordot/GatherV2GatherV2>model_12/transformer_encoder_23/dense/Tensordot/Shape:output:0=model_12/transformer_encoder_23/dense/Tensordot/free:output:0Fmodel_12/transformer_encoder_23/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
?model_12/transformer_encoder_23/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_12/transformer_encoder_23/dense/Tensordot/GatherV2_1GatherV2>model_12/transformer_encoder_23/dense/Tensordot/Shape:output:0=model_12/transformer_encoder_23/dense/Tensordot/axes:output:0Hmodel_12/transformer_encoder_23/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5model_12/transformer_encoder_23/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4model_12/transformer_encoder_23/dense/Tensordot/ProdProdAmodel_12/transformer_encoder_23/dense/Tensordot/GatherV2:output:0>model_12/transformer_encoder_23/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
7model_12/transformer_encoder_23/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
6model_12/transformer_encoder_23/dense/Tensordot/Prod_1ProdCmodel_12/transformer_encoder_23/dense/Tensordot/GatherV2_1:output:0@model_12/transformer_encoder_23/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: }
;model_12/transformer_encoder_23/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_12/transformer_encoder_23/dense/Tensordot/concatConcatV2=model_12/transformer_encoder_23/dense/Tensordot/free:output:0=model_12/transformer_encoder_23/dense/Tensordot/axes:output:0Dmodel_12/transformer_encoder_23/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5model_12/transformer_encoder_23/dense/Tensordot/stackPack=model_12/transformer_encoder_23/dense/Tensordot/Prod:output:0?model_12/transformer_encoder_23/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
9model_12/transformer_encoder_23/dense/Tensordot/transpose	TransposeGmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add_1:z:0?model_12/transformer_encoder_23/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
7model_12/transformer_encoder_23/dense/Tensordot/ReshapeReshape=model_12/transformer_encoder_23/dense/Tensordot/transpose:y:0>model_12/transformer_encoder_23/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
6model_12/transformer_encoder_23/dense/Tensordot/MatMulMatMul@model_12/transformer_encoder_23/dense/Tensordot/Reshape:output:0Fmodel_12/transformer_encoder_23/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
7model_12/transformer_encoder_23/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
=model_12/transformer_encoder_23/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_12/transformer_encoder_23/dense/Tensordot/concat_1ConcatV2Amodel_12/transformer_encoder_23/dense/Tensordot/GatherV2:output:0@model_12/transformer_encoder_23/dense/Tensordot/Const_2:output:0Fmodel_12/transformer_encoder_23/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
/model_12/transformer_encoder_23/dense/TensordotReshape@model_12/transformer_encoder_23/dense/Tensordot/MatMul:product:0Amodel_12/transformer_encoder_23/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
<model_12/transformer_encoder_23/dense/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_transformer_encoder_23_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
-model_12/transformer_encoder_23/dense/BiasAddBiasAdd8model_12/transformer_encoder_23/dense/Tensordot:output:0Dmodel_12/transformer_encoder_23/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
*model_12/transformer_encoder_23/dense/ReluRelu6model_12/transformer_encoder_23/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
@model_12/transformer_encoder_23/dense_1/Tensordot/ReadVariableOpReadVariableOpImodel_12_transformer_encoder_23_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0?
6model_12/transformer_encoder_23/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
6model_12/transformer_encoder_23/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
7model_12/transformer_encoder_23/dense_1/Tensordot/ShapeShape8model_12/transformer_encoder_23/dense/Relu:activations:0*
T0*
_output_shapes
:?
?model_12/transformer_encoder_23/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_12/transformer_encoder_23/dense_1/Tensordot/GatherV2GatherV2@model_12/transformer_encoder_23/dense_1/Tensordot/Shape:output:0?model_12/transformer_encoder_23/dense_1/Tensordot/free:output:0Hmodel_12/transformer_encoder_23/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Amodel_12/transformer_encoder_23/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<model_12/transformer_encoder_23/dense_1/Tensordot/GatherV2_1GatherV2@model_12/transformer_encoder_23/dense_1/Tensordot/Shape:output:0?model_12/transformer_encoder_23/dense_1/Tensordot/axes:output:0Jmodel_12/transformer_encoder_23/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
7model_12/transformer_encoder_23/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
6model_12/transformer_encoder_23/dense_1/Tensordot/ProdProdCmodel_12/transformer_encoder_23/dense_1/Tensordot/GatherV2:output:0@model_12/transformer_encoder_23/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
9model_12/transformer_encoder_23/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
8model_12/transformer_encoder_23/dense_1/Tensordot/Prod_1ProdEmodel_12/transformer_encoder_23/dense_1/Tensordot/GatherV2_1:output:0Bmodel_12/transformer_encoder_23/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
=model_12/transformer_encoder_23/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_12/transformer_encoder_23/dense_1/Tensordot/concatConcatV2?model_12/transformer_encoder_23/dense_1/Tensordot/free:output:0?model_12/transformer_encoder_23/dense_1/Tensordot/axes:output:0Fmodel_12/transformer_encoder_23/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7model_12/transformer_encoder_23/dense_1/Tensordot/stackPack?model_12/transformer_encoder_23/dense_1/Tensordot/Prod:output:0Amodel_12/transformer_encoder_23/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
;model_12/transformer_encoder_23/dense_1/Tensordot/transpose	Transpose8model_12/transformer_encoder_23/dense/Relu:activations:0Amodel_12/transformer_encoder_23/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
9model_12/transformer_encoder_23/dense_1/Tensordot/ReshapeReshape?model_12/transformer_encoder_23/dense_1/Tensordot/transpose:y:0@model_12/transformer_encoder_23/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
8model_12/transformer_encoder_23/dense_1/Tensordot/MatMulMatMulBmodel_12/transformer_encoder_23/dense_1/Tensordot/Reshape:output:0Hmodel_12/transformer_encoder_23/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9model_12/transformer_encoder_23/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
?model_12/transformer_encoder_23/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_12/transformer_encoder_23/dense_1/Tensordot/concat_1ConcatV2Cmodel_12/transformer_encoder_23/dense_1/Tensordot/GatherV2:output:0Bmodel_12/transformer_encoder_23/dense_1/Tensordot/Const_2:output:0Hmodel_12/transformer_encoder_23/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
1model_12/transformer_encoder_23/dense_1/TensordotReshapeBmodel_12/transformer_encoder_23/dense_1/Tensordot/MatMul:product:0Cmodel_12/transformer_encoder_23/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
>model_12/transformer_encoder_23/dense_1/BiasAdd/ReadVariableOpReadVariableOpGmodel_12_transformer_encoder_23_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
/model_12/transformer_encoder_23/dense_1/BiasAddBiasAdd:model_12/transformer_encoder_23/dense_1/Tensordot:output:0Fmodel_12/transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
2model_12/transformer_encoder_23/dropout_1/IdentityIdentity8model_12/transformer_encoder_23/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
%model_12/transformer_encoder_23/add_1AddV2Gmodel_12/transformer_encoder_23/layer_normalization/batchnorm/add_1:z:0;model_12/transformer_encoder_23/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Tmodel_12/transformer_encoder_23/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_12/transformer_encoder_23/layer_normalization_1/moments/meanMean)model_12/transformer_encoder_23/add_1:z:0]model_12/transformer_encoder_23/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Jmodel_12/transformer_encoder_23/layer_normalization_1/moments/StopGradientStopGradientKmodel_12/transformer_encoder_23/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Omodel_12/transformer_encoder_23/layer_normalization_1/moments/SquaredDifferenceSquaredDifference)model_12/transformer_encoder_23/add_1:z:0Smodel_12/transformer_encoder_23/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Xmodel_12/transformer_encoder_23/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Fmodel_12/transformer_encoder_23/layer_normalization_1/moments/varianceMeanSmodel_12/transformer_encoder_23/layer_normalization_1/moments/SquaredDifference:z:0amodel_12/transformer_encoder_23/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Emodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Cmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/addAddV2Omodel_12/transformer_encoder_23/layer_normalization_1/moments/variance:output:0Nmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/RsqrtRsqrtGmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Rmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp[model_12_transformer_encoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mulMulImodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/Rsqrt:y:0Zmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul_1Mul)model_12/transformer_encoder_23/add_1:z:0Gmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul_2MulKmodel_12/transformer_encoder_23/layer_normalization_1/moments/mean:output:0Gmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Nmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpWmodel_12_transformer_encoder_23_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/subSubVmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp:value:0Imodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add_1AddV2Imodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul_1:z:0Gmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
%model_12/transformer_decoder_23/ShapeShapeImodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:}
3model_12/transformer_decoder_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5model_12/transformer_decoder_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5model_12/transformer_decoder_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-model_12/transformer_decoder_23/strided_sliceStridedSlice.model_12/transformer_decoder_23/Shape:output:0<model_12/transformer_decoder_23/strided_slice/stack:output:0>model_12/transformer_decoder_23/strided_slice/stack_1:output:0>model_12/transformer_decoder_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5model_12/transformer_decoder_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7model_12/transformer_decoder_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7model_12/transformer_decoder_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/model_12/transformer_decoder_23/strided_slice_1StridedSlice.model_12/transformer_decoder_23/Shape:output:0>model_12/transformer_decoder_23/strided_slice_1/stack:output:0@model_12/transformer_decoder_23/strided_slice_1/stack_1:output:0@model_12/transformer_decoder_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+model_12/transformer_decoder_23/range/startConst*
_output_shapes
: *
dtype0*
value	B : m
+model_12/transformer_decoder_23/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
%model_12/transformer_decoder_23/rangeRange4model_12/transformer_decoder_23/range/start:output:08model_12/transformer_decoder_23/strided_slice_1:output:04model_12/transformer_decoder_23/range/delta:output:0*
_output_shapes
: ?
5model_12/transformer_decoder_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7model_12/transformer_decoder_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
7model_12/transformer_decoder_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/model_12/transformer_decoder_23/strided_slice_2StridedSlice.model_12/transformer_decoder_23/range:output:0>model_12/transformer_decoder_23/strided_slice_2/stack:output:0@model_12/transformer_decoder_23/strided_slice_2/stack_1:output:0@model_12/transformer_decoder_23/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_masko
-model_12/transformer_decoder_23/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : o
-model_12/transformer_decoder_23/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
'model_12/transformer_decoder_23/range_1Range6model_12/transformer_decoder_23/range_1/start:output:08model_12/transformer_decoder_23/strided_slice_1:output:06model_12/transformer_decoder_23/range_1/delta:output:0*
_output_shapes
: ?
,model_12/transformer_decoder_23/GreaterEqualGreaterEqual8model_12/transformer_decoder_23/strided_slice_2:output:00model_12/transformer_decoder_23/range_1:output:0*
T0*
_output_shapes

:  ?
$model_12/transformer_decoder_23/CastCast0model_12/transformer_decoder_23/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  
5model_12/transformer_decoder_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7model_12/transformer_decoder_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7model_12/transformer_decoder_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/model_12/transformer_decoder_23/strided_slice_3StridedSlice.model_12/transformer_decoder_23/Shape:output:0>model_12/transformer_decoder_23/strided_slice_3/stack:output:0@model_12/transformer_decoder_23/strided_slice_3/stack_1:output:0@model_12/transformer_decoder_23/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5model_12/transformer_decoder_23/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
7model_12/transformer_decoder_23/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7model_12/transformer_decoder_23/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/model_12/transformer_decoder_23/strided_slice_4StridedSlice.model_12/transformer_decoder_23/Shape:output:0>model_12/transformer_decoder_23/strided_slice_4/stack:output:0@model_12/transformer_decoder_23/strided_slice_4/stack_1:output:0@model_12/transformer_decoder_23/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/model_12/transformer_decoder_23/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
-model_12/transformer_decoder_23/Reshape/shapePack8model_12/transformer_decoder_23/Reshape/shape/0:output:08model_12/transformer_decoder_23/strided_slice_3:output:08model_12/transformer_decoder_23/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
'model_12/transformer_decoder_23/ReshapeReshape(model_12/transformer_decoder_23/Cast:y:06model_12/transformer_decoder_23/Reshape/shape:output:0*
T0*"
_output_shapes
:  y
.model_12/transformer_decoder_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
*model_12/transformer_decoder_23/ExpandDims
ExpandDims6model_12/transformer_decoder_23/strided_slice:output:07model_12/transformer_decoder_23/ExpandDims/dim:output:0*
T0*
_output_shapes
:v
%model_12/transformer_decoder_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"      m
+model_12/transformer_decoder_23/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
&model_12/transformer_decoder_23/concatConcatV23model_12/transformer_decoder_23/ExpandDims:output:0.model_12/transformer_decoder_23/Const:output:04model_12/transformer_decoder_23/concat/axis:output:0*
N*
T0*
_output_shapes
:?
$model_12/transformer_decoder_23/TileTile0model_12/transformer_decoder_23/Reshape:output:0/model_12/transformer_decoder_23/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Wmodel_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp`model_12_transformer_decoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Hmodel_12/transformer_decoder_23/multi_head_attention/query/einsum/EinsumEinsumImodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0_model_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Mmodel_12/transformer_decoder_23/multi_head_attention/query/add/ReadVariableOpReadVariableOpVmodel_12_transformer_decoder_23_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
>model_12/transformer_decoder_23/multi_head_attention/query/addAddV2Qmodel_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum:output:0Umodel_12/transformer_decoder_23/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Umodel_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp^model_12_transformer_decoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fmodel_12/transformer_decoder_23/multi_head_attention/key/einsum/EinsumEinsumImodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0]model_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Kmodel_12/transformer_decoder_23/multi_head_attention/key/add/ReadVariableOpReadVariableOpTmodel_12_transformer_decoder_23_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
<model_12/transformer_decoder_23/multi_head_attention/key/addAddV2Omodel_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum:output:0Smodel_12/transformer_decoder_23/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Wmodel_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp`model_12_transformer_decoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Hmodel_12/transformer_decoder_23/multi_head_attention/value/einsum/EinsumEinsumImodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0_model_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Mmodel_12/transformer_decoder_23/multi_head_attention/value/add/ReadVariableOpReadVariableOpVmodel_12_transformer_decoder_23_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
>model_12/transformer_decoder_23/multi_head_attention/value/addAddV2Qmodel_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum:output:0Umodel_12/transformer_decoder_23/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 
:model_12/transformer_decoder_23/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
8model_12/transformer_decoder_23/multi_head_attention/MulMulBmodel_12/transformer_decoder_23/multi_head_attention/query/add:z:0Cmodel_12/transformer_decoder_23/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
Bmodel_12/transformer_decoder_23/multi_head_attention/einsum/EinsumEinsum@model_12/transformer_decoder_23/multi_head_attention/key/add:z:0<model_12/transformer_decoder_23/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
Cmodel_12/transformer_decoder_23/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
?model_12/transformer_decoder_23/multi_head_attention/ExpandDims
ExpandDims-model_12/transformer_decoder_23/Tile:output:0Lmodel_12/transformer_decoder_23/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
Amodel_12/transformer_decoder_23/multi_head_attention/softmax/CastCastHmodel_12/transformer_decoder_23/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ?
Bmodel_12/transformer_decoder_23/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
@model_12/transformer_decoder_23/multi_head_attention/softmax/subSubKmodel_12/transformer_decoder_23/multi_head_attention/softmax/sub/x:output:0Emodel_12/transformer_decoder_23/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
Bmodel_12/transformer_decoder_23/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
@model_12/transformer_decoder_23/multi_head_attention/softmax/mulMulDmodel_12/transformer_decoder_23/multi_head_attention/softmax/sub:z:0Kmodel_12/transformer_decoder_23/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
@model_12/transformer_decoder_23/multi_head_attention/softmax/addAddV2Kmodel_12/transformer_decoder_23/multi_head_attention/einsum/Einsum:output:0Dmodel_12/transformer_decoder_23/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
Dmodel_12/transformer_decoder_23/multi_head_attention/softmax/SoftmaxSoftmaxDmodel_12/transformer_decoder_23/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
Gmodel_12/transformer_decoder_23/multi_head_attention/dropout_2/IdentityIdentityNmodel_12/transformer_decoder_23/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
Dmodel_12/transformer_decoder_23/multi_head_attention/einsum_1/EinsumEinsumPmodel_12/transformer_decoder_23/multi_head_attention/dropout_2/Identity:output:0Bmodel_12/transformer_decoder_23/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
bmodel_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpkmodel_12_transformer_decoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Smodel_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/EinsumEinsumMmodel_12/transformer_decoder_23/multi_head_attention/einsum_1/Einsum:output:0jmodel_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Xmodel_12/transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpamodel_12_transformer_decoder_23_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
Imodel_12/transformer_decoder_23/multi_head_attention/attention_output/addAddV2\model_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum:output:0`model_12/transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
0model_12/transformer_decoder_23/dropout/IdentityIdentityMmodel_12/transformer_decoder_23/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
#model_12/transformer_decoder_23/addAddV29model_12/transformer_decoder_23/dropout/Identity:output:0Imodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ?
Rmodel_12/transformer_decoder_23/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_12/transformer_decoder_23/layer_normalization/moments/meanMean'model_12/transformer_decoder_23/add:z:0[model_12/transformer_decoder_23/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Hmodel_12/transformer_decoder_23/layer_normalization/moments/StopGradientStopGradientImodel_12/transformer_decoder_23/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Mmodel_12/transformer_decoder_23/layer_normalization/moments/SquaredDifferenceSquaredDifference'model_12/transformer_decoder_23/add:z:0Qmodel_12/transformer_decoder_23/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Vmodel_12/transformer_decoder_23/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_12/transformer_decoder_23/layer_normalization/moments/varianceMeanQmodel_12/transformer_decoder_23/layer_normalization/moments/SquaredDifference:z:0_model_12/transformer_decoder_23/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Cmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Amodel_12/transformer_decoder_23/layer_normalization/batchnorm/addAddV2Mmodel_12/transformer_decoder_23/layer_normalization/moments/variance:output:0Lmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_decoder_23/layer_normalization/batchnorm/RsqrtRsqrtEmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Pmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_12_transformer_decoder_23_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_12/transformer_decoder_23/layer_normalization/batchnorm/mulMulGmodel_12/transformer_decoder_23/layer_normalization/batchnorm/Rsqrt:y:0Xmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul_1Mul'model_12/transformer_decoder_23/add:z:0Emodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul_2MulImodel_12/transformer_decoder_23/layer_normalization/moments/mean:output:0Emodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Lmodel_12/transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpReadVariableOpUmodel_12_transformer_decoder_23_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_12/transformer_decoder_23/layer_normalization/batchnorm/subSubTmodel_12/transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp:value:0Gmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add_1AddV2Gmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul_1:z:0Emodel_12/transformer_decoder_23/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
>model_12/transformer_decoder_23/dense/Tensordot/ReadVariableOpReadVariableOpGmodel_12_transformer_decoder_23_dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0~
4model_12/transformer_decoder_23/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
4model_12/transformer_decoder_23/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
5model_12/transformer_decoder_23/dense/Tensordot/ShapeShapeGmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
=model_12/transformer_decoder_23/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_12/transformer_decoder_23/dense/Tensordot/GatherV2GatherV2>model_12/transformer_decoder_23/dense/Tensordot/Shape:output:0=model_12/transformer_decoder_23/dense/Tensordot/free:output:0Fmodel_12/transformer_decoder_23/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
?model_12/transformer_decoder_23/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_12/transformer_decoder_23/dense/Tensordot/GatherV2_1GatherV2>model_12/transformer_decoder_23/dense/Tensordot/Shape:output:0=model_12/transformer_decoder_23/dense/Tensordot/axes:output:0Hmodel_12/transformer_decoder_23/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5model_12/transformer_decoder_23/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4model_12/transformer_decoder_23/dense/Tensordot/ProdProdAmodel_12/transformer_decoder_23/dense/Tensordot/GatherV2:output:0>model_12/transformer_decoder_23/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
7model_12/transformer_decoder_23/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
6model_12/transformer_decoder_23/dense/Tensordot/Prod_1ProdCmodel_12/transformer_decoder_23/dense/Tensordot/GatherV2_1:output:0@model_12/transformer_decoder_23/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: }
;model_12/transformer_decoder_23/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_12/transformer_decoder_23/dense/Tensordot/concatConcatV2=model_12/transformer_decoder_23/dense/Tensordot/free:output:0=model_12/transformer_decoder_23/dense/Tensordot/axes:output:0Dmodel_12/transformer_decoder_23/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5model_12/transformer_decoder_23/dense/Tensordot/stackPack=model_12/transformer_decoder_23/dense/Tensordot/Prod:output:0?model_12/transformer_decoder_23/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
9model_12/transformer_decoder_23/dense/Tensordot/transpose	TransposeGmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add_1:z:0?model_12/transformer_decoder_23/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
7model_12/transformer_decoder_23/dense/Tensordot/ReshapeReshape=model_12/transformer_decoder_23/dense/Tensordot/transpose:y:0>model_12/transformer_decoder_23/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
6model_12/transformer_decoder_23/dense/Tensordot/MatMulMatMul@model_12/transformer_decoder_23/dense/Tensordot/Reshape:output:0Fmodel_12/transformer_decoder_23/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
7model_12/transformer_decoder_23/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
=model_12/transformer_decoder_23/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_12/transformer_decoder_23/dense/Tensordot/concat_1ConcatV2Amodel_12/transformer_decoder_23/dense/Tensordot/GatherV2:output:0@model_12/transformer_decoder_23/dense/Tensordot/Const_2:output:0Fmodel_12/transformer_decoder_23/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
/model_12/transformer_decoder_23/dense/TensordotReshape@model_12/transformer_decoder_23/dense/Tensordot/MatMul:product:0Amodel_12/transformer_decoder_23/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
<model_12/transformer_decoder_23/dense/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_transformer_decoder_23_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
-model_12/transformer_decoder_23/dense/BiasAddBiasAdd8model_12/transformer_decoder_23/dense/Tensordot:output:0Dmodel_12/transformer_decoder_23/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
*model_12/transformer_decoder_23/dense/ReluRelu6model_12/transformer_decoder_23/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
@model_12/transformer_decoder_23/dense_1/Tensordot/ReadVariableOpReadVariableOpImodel_12_transformer_decoder_23_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0?
6model_12/transformer_decoder_23/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
6model_12/transformer_decoder_23/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
7model_12/transformer_decoder_23/dense_1/Tensordot/ShapeShape8model_12/transformer_decoder_23/dense/Relu:activations:0*
T0*
_output_shapes
:?
?model_12/transformer_decoder_23/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_12/transformer_decoder_23/dense_1/Tensordot/GatherV2GatherV2@model_12/transformer_decoder_23/dense_1/Tensordot/Shape:output:0?model_12/transformer_decoder_23/dense_1/Tensordot/free:output:0Hmodel_12/transformer_decoder_23/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
Amodel_12/transformer_decoder_23/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
<model_12/transformer_decoder_23/dense_1/Tensordot/GatherV2_1GatherV2@model_12/transformer_decoder_23/dense_1/Tensordot/Shape:output:0?model_12/transformer_decoder_23/dense_1/Tensordot/axes:output:0Jmodel_12/transformer_decoder_23/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
7model_12/transformer_decoder_23/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
6model_12/transformer_decoder_23/dense_1/Tensordot/ProdProdCmodel_12/transformer_decoder_23/dense_1/Tensordot/GatherV2:output:0@model_12/transformer_decoder_23/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
9model_12/transformer_decoder_23/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
8model_12/transformer_decoder_23/dense_1/Tensordot/Prod_1ProdEmodel_12/transformer_decoder_23/dense_1/Tensordot/GatherV2_1:output:0Bmodel_12/transformer_decoder_23/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
=model_12/transformer_decoder_23/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_12/transformer_decoder_23/dense_1/Tensordot/concatConcatV2?model_12/transformer_decoder_23/dense_1/Tensordot/free:output:0?model_12/transformer_decoder_23/dense_1/Tensordot/axes:output:0Fmodel_12/transformer_decoder_23/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
7model_12/transformer_decoder_23/dense_1/Tensordot/stackPack?model_12/transformer_decoder_23/dense_1/Tensordot/Prod:output:0Amodel_12/transformer_decoder_23/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
;model_12/transformer_decoder_23/dense_1/Tensordot/transpose	Transpose8model_12/transformer_decoder_23/dense/Relu:activations:0Amodel_12/transformer_decoder_23/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
9model_12/transformer_decoder_23/dense_1/Tensordot/ReshapeReshape?model_12/transformer_decoder_23/dense_1/Tensordot/transpose:y:0@model_12/transformer_decoder_23/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
8model_12/transformer_decoder_23/dense_1/Tensordot/MatMulMatMulBmodel_12/transformer_decoder_23/dense_1/Tensordot/Reshape:output:0Hmodel_12/transformer_decoder_23/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
9model_12/transformer_decoder_23/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?
?model_12/transformer_decoder_23/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_12/transformer_decoder_23/dense_1/Tensordot/concat_1ConcatV2Cmodel_12/transformer_decoder_23/dense_1/Tensordot/GatherV2:output:0Bmodel_12/transformer_decoder_23/dense_1/Tensordot/Const_2:output:0Hmodel_12/transformer_decoder_23/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
1model_12/transformer_decoder_23/dense_1/TensordotReshapeBmodel_12/transformer_decoder_23/dense_1/Tensordot/MatMul:product:0Cmodel_12/transformer_decoder_23/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
>model_12/transformer_decoder_23/dense_1/BiasAdd/ReadVariableOpReadVariableOpGmodel_12_transformer_decoder_23_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
/model_12/transformer_decoder_23/dense_1/BiasAddBiasAdd:model_12/transformer_decoder_23/dense_1/Tensordot:output:0Fmodel_12/transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
2model_12/transformer_decoder_23/dropout_1/IdentityIdentity8model_12/transformer_decoder_23/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
%model_12/transformer_decoder_23/add_1AddV2Gmodel_12/transformer_decoder_23/layer_normalization/batchnorm/add_1:z:0;model_12/transformer_decoder_23/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Tmodel_12/transformer_decoder_23/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_12/transformer_decoder_23/layer_normalization_1/moments/meanMean)model_12/transformer_decoder_23/add_1:z:0]model_12/transformer_decoder_23/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Jmodel_12/transformer_decoder_23/layer_normalization_1/moments/StopGradientStopGradientKmodel_12/transformer_decoder_23/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Omodel_12/transformer_decoder_23/layer_normalization_1/moments/SquaredDifferenceSquaredDifference)model_12/transformer_decoder_23/add_1:z:0Smodel_12/transformer_decoder_23/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Xmodel_12/transformer_decoder_23/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Fmodel_12/transformer_decoder_23/layer_normalization_1/moments/varianceMeanSmodel_12/transformer_decoder_23/layer_normalization_1/moments/SquaredDifference:z:0amodel_12/transformer_decoder_23/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Emodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Cmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/addAddV2Omodel_12/transformer_decoder_23/layer_normalization_1/moments/variance:output:0Nmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/RsqrtRsqrtGmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Rmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp[model_12_transformer_decoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mulMulImodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/Rsqrt:y:0Zmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul_1Mul)model_12/transformer_decoder_23/add_1:z:0Gmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul_2MulKmodel_12/transformer_decoder_23/layer_normalization_1/moments/mean:output:0Gmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Nmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpWmodel_12_transformer_decoder_23_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
Cmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/subSubVmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp:value:0Imodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Emodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/add_1AddV2Imodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul_1:z:0Gmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? }
;model_12/global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
)model_12/global_average_pooling1d_12/MeanMeanImodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/add_1:z:0Dmodel_12/global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
'model_12/dense_23/MatMul/ReadVariableOpReadVariableOp0model_12_dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model_12/dense_23/MatMulMatMul2model_12/global_average_pooling1d_12/Mean:output:0/model_12/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
(model_12/dense_23/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_12/dense_23/BiasAddBiasAdd"model_12/dense_23/MatMul:product:00model_12/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@t
model_12/dense_23/ReluRelu"model_12/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
model_12/dropout_10/IdentityIdentity$model_12/dense_23/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
'model_12/dense_24/MatMul/ReadVariableOpReadVariableOp0model_12_dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model_12/dense_24/MatMulMatMul%model_12/dropout_10/Identity:output:0/model_12/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(model_12/dense_24/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_12/dense_24/BiasAddBiasAdd"model_12/dense_24/MatMul:product:00model_12/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
model_12/dense_24/SigmoidSigmoid"model_12/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitymodel_12/dense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^model_12/dense_23/BiasAdd/ReadVariableOp(^model_12/dense_23/MatMul/ReadVariableOp)^model_12/dense_24/BiasAdd/ReadVariableOp(^model_12/dense_24/MatMul/ReadVariableOpN^model_12/text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2M^model_12/token_and_position_embedding_27/position_embedding28/ReadVariableOpL^model_12/token_and_position_embedding_27/token_embedding28/embedding_lookupM^model_12/token_and_position_embedding_28/position_embedding29/ReadVariableOpL^model_12/token_and_position_embedding_28/token_embedding29/embedding_lookup=^model_12/transformer_decoder_23/dense/BiasAdd/ReadVariableOp?^model_12/transformer_decoder_23/dense/Tensordot/ReadVariableOp?^model_12/transformer_decoder_23/dense_1/BiasAdd/ReadVariableOpA^model_12/transformer_decoder_23/dense_1/Tensordot/ReadVariableOpM^model_12/transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpQ^model_12/transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpO^model_12/transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpS^model_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpY^model_12/transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpc^model_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpL^model_12/transformer_decoder_23/multi_head_attention/key/add/ReadVariableOpV^model_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpN^model_12/transformer_decoder_23/multi_head_attention/query/add/ReadVariableOpX^model_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpN^model_12/transformer_decoder_23/multi_head_attention/value/add/ReadVariableOpX^model_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp=^model_12/transformer_encoder_23/dense/BiasAdd/ReadVariableOp?^model_12/transformer_encoder_23/dense/Tensordot/ReadVariableOp?^model_12/transformer_encoder_23/dense_1/BiasAdd/ReadVariableOpA^model_12/transformer_encoder_23/dense_1/Tensordot/ReadVariableOpM^model_12/transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpQ^model_12/transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpO^model_12/transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpS^model_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpY^model_12/transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpc^model_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpL^model_12/transformer_encoder_23/multi_head_attention/key/add/ReadVariableOpV^model_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpN^model_12/transformer_encoder_23/multi_head_attention/query/add/ReadVariableOpX^model_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpN^model_12/transformer_encoder_23/multi_head_attention/value/add/ReadVariableOpX^model_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(model_12/dense_23/BiasAdd/ReadVariableOp(model_12/dense_23/BiasAdd/ReadVariableOp2R
'model_12/dense_23/MatMul/ReadVariableOp'model_12/dense_23/MatMul/ReadVariableOp2T
(model_12/dense_24/BiasAdd/ReadVariableOp(model_12/dense_24/BiasAdd/ReadVariableOp2R
'model_12/dense_24/MatMul/ReadVariableOp'model_12/dense_24/MatMul/ReadVariableOp2?
Mmodel_12/text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Mmodel_12/text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22?
Lmodel_12/token_and_position_embedding_27/position_embedding28/ReadVariableOpLmodel_12/token_and_position_embedding_27/position_embedding28/ReadVariableOp2?
Kmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookupKmodel_12/token_and_position_embedding_27/token_embedding28/embedding_lookup2?
Lmodel_12/token_and_position_embedding_28/position_embedding29/ReadVariableOpLmodel_12/token_and_position_embedding_28/position_embedding29/ReadVariableOp2?
Kmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookupKmodel_12/token_and_position_embedding_28/token_embedding29/embedding_lookup2|
<model_12/transformer_decoder_23/dense/BiasAdd/ReadVariableOp<model_12/transformer_decoder_23/dense/BiasAdd/ReadVariableOp2?
>model_12/transformer_decoder_23/dense/Tensordot/ReadVariableOp>model_12/transformer_decoder_23/dense/Tensordot/ReadVariableOp2?
>model_12/transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp>model_12/transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp2?
@model_12/transformer_decoder_23/dense_1/Tensordot/ReadVariableOp@model_12/transformer_decoder_23/dense_1/Tensordot/ReadVariableOp2?
Lmodel_12/transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpLmodel_12/transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp2?
Pmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpPmodel_12/transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp2?
Nmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpNmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp2?
Rmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpRmodel_12/transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Xmodel_12/transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpXmodel_12/transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp2?
bmodel_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpbmodel_12/transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Kmodel_12/transformer_decoder_23/multi_head_attention/key/add/ReadVariableOpKmodel_12/transformer_decoder_23/multi_head_attention/key/add/ReadVariableOp2?
Umodel_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpUmodel_12/transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Mmodel_12/transformer_decoder_23/multi_head_attention/query/add/ReadVariableOpMmodel_12/transformer_decoder_23/multi_head_attention/query/add/ReadVariableOp2?
Wmodel_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpWmodel_12/transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Mmodel_12/transformer_decoder_23/multi_head_attention/value/add/ReadVariableOpMmodel_12/transformer_decoder_23/multi_head_attention/value/add/ReadVariableOp2?
Wmodel_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpWmodel_12/transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp2|
<model_12/transformer_encoder_23/dense/BiasAdd/ReadVariableOp<model_12/transformer_encoder_23/dense/BiasAdd/ReadVariableOp2?
>model_12/transformer_encoder_23/dense/Tensordot/ReadVariableOp>model_12/transformer_encoder_23/dense/Tensordot/ReadVariableOp2?
>model_12/transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp>model_12/transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp2?
@model_12/transformer_encoder_23/dense_1/Tensordot/ReadVariableOp@model_12/transformer_encoder_23/dense_1/Tensordot/ReadVariableOp2?
Lmodel_12/transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpLmodel_12/transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp2?
Pmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpPmodel_12/transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp2?
Nmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpNmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp2?
Rmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpRmodel_12/transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Xmodel_12/transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpXmodel_12/transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp2?
bmodel_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpbmodel_12/transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Kmodel_12/transformer_encoder_23/multi_head_attention/key/add/ReadVariableOpKmodel_12/transformer_encoder_23/multi_head_attention/key/add/ReadVariableOp2?
Umodel_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpUmodel_12/transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Mmodel_12/transformer_encoder_23/multi_head_attention/query/add/ReadVariableOpMmodel_12/transformer_encoder_23/multi_head_attention/query/add/ReadVariableOp2?
Wmodel_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpWmodel_12/transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Mmodel_12/transformer_encoder_23/multi_head_attention/value/add/ReadVariableOpMmodel_12/transformer_encoder_23/multi_head_attention/value/add/ReadVariableOp2?
Wmodel_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpWmodel_12/transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namephrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
token_role:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?

)__inference_model_12_layer_call_fn_764209
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:

unknown_37:

unknown_38:

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_763402o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
٣
?T
__inference__traced_save_766517
file_prefix.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop[
Wsavev2_token_and_position_embedding_27_token_embedding28_embeddings_read_readvariableop^
Zsavev2_token_and_position_embedding_27_position_embedding28_embeddings_read_readvariableop[
Wsavev2_token_and_position_embedding_28_token_embedding29_embeddings_read_readvariableop^
Zsavev2_token_and_position_embedding_28_position_embedding29_embeddings_read_readvariableopW
Ssavev2_transformer_encoder_23_multi_head_attention_query_kernel_read_readvariableopU
Qsavev2_transformer_encoder_23_multi_head_attention_query_bias_read_readvariableopU
Qsavev2_transformer_encoder_23_multi_head_attention_key_kernel_read_readvariableopS
Osavev2_transformer_encoder_23_multi_head_attention_key_bias_read_readvariableopW
Ssavev2_transformer_encoder_23_multi_head_attention_value_kernel_read_readvariableopU
Qsavev2_transformer_encoder_23_multi_head_attention_value_bias_read_readvariableopb
^savev2_transformer_encoder_23_multi_head_attention_attention_output_kernel_read_readvariableop`
\savev2_transformer_encoder_23_multi_head_attention_attention_output_bias_read_readvariableopO
Ksavev2_transformer_encoder_23_layer_normalization_gamma_read_readvariableopN
Jsavev2_transformer_encoder_23_layer_normalization_beta_read_readvariableopQ
Msavev2_transformer_encoder_23_layer_normalization_1_gamma_read_readvariableopP
Lsavev2_transformer_encoder_23_layer_normalization_1_beta_read_readvariableopB
>savev2_transformer_encoder_23_dense_kernel_read_readvariableop@
<savev2_transformer_encoder_23_dense_bias_read_readvariableopD
@savev2_transformer_encoder_23_dense_1_kernel_read_readvariableopB
>savev2_transformer_encoder_23_dense_1_bias_read_readvariableopW
Ssavev2_transformer_decoder_23_multi_head_attention_query_kernel_read_readvariableopU
Qsavev2_transformer_decoder_23_multi_head_attention_query_bias_read_readvariableopU
Qsavev2_transformer_decoder_23_multi_head_attention_key_kernel_read_readvariableopS
Osavev2_transformer_decoder_23_multi_head_attention_key_bias_read_readvariableopW
Ssavev2_transformer_decoder_23_multi_head_attention_value_kernel_read_readvariableopU
Qsavev2_transformer_decoder_23_multi_head_attention_value_bias_read_readvariableopb
^savev2_transformer_decoder_23_multi_head_attention_attention_output_kernel_read_readvariableop`
\savev2_transformer_decoder_23_multi_head_attention_attention_output_bias_read_readvariableopO
Ksavev2_transformer_decoder_23_layer_normalization_gamma_read_readvariableopN
Jsavev2_transformer_decoder_23_layer_normalization_beta_read_readvariableopQ
Msavev2_transformer_decoder_23_layer_normalization_1_gamma_read_readvariableopP
Lsavev2_transformer_decoder_23_layer_normalization_1_beta_read_readvariableopB
>savev2_transformer_decoder_23_dense_kernel_read_readvariableop@
<savev2_transformer_decoder_23_dense_bias_read_readvariableopD
@savev2_transformer_decoder_23_dense_1_kernel_read_readvariableopB
>savev2_transformer_decoder_23_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableopb
^savev2_adam_token_and_position_embedding_27_token_embedding28_embeddings_m_read_readvariableope
asavev2_adam_token_and_position_embedding_27_position_embedding28_embeddings_m_read_readvariableopb
^savev2_adam_token_and_position_embedding_28_token_embedding29_embeddings_m_read_readvariableope
asavev2_adam_token_and_position_embedding_28_position_embedding29_embeddings_m_read_readvariableop^
Zsavev2_adam_transformer_encoder_23_multi_head_attention_query_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_encoder_23_multi_head_attention_query_bias_m_read_readvariableop\
Xsavev2_adam_transformer_encoder_23_multi_head_attention_key_kernel_m_read_readvariableopZ
Vsavev2_adam_transformer_encoder_23_multi_head_attention_key_bias_m_read_readvariableop^
Zsavev2_adam_transformer_encoder_23_multi_head_attention_value_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_encoder_23_multi_head_attention_value_bias_m_read_readvariableopi
esavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_m_read_readvariableopg
csavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_m_read_readvariableopV
Rsavev2_adam_transformer_encoder_23_layer_normalization_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_encoder_23_layer_normalization_beta_m_read_readvariableopX
Tsavev2_adam_transformer_encoder_23_layer_normalization_1_gamma_m_read_readvariableopW
Ssavev2_adam_transformer_encoder_23_layer_normalization_1_beta_m_read_readvariableopI
Esavev2_adam_transformer_encoder_23_dense_kernel_m_read_readvariableopG
Csavev2_adam_transformer_encoder_23_dense_bias_m_read_readvariableopK
Gsavev2_adam_transformer_encoder_23_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_transformer_encoder_23_dense_1_bias_m_read_readvariableop^
Zsavev2_adam_transformer_decoder_23_multi_head_attention_query_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_decoder_23_multi_head_attention_query_bias_m_read_readvariableop\
Xsavev2_adam_transformer_decoder_23_multi_head_attention_key_kernel_m_read_readvariableopZ
Vsavev2_adam_transformer_decoder_23_multi_head_attention_key_bias_m_read_readvariableop^
Zsavev2_adam_transformer_decoder_23_multi_head_attention_value_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_decoder_23_multi_head_attention_value_bias_m_read_readvariableopi
esavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_m_read_readvariableopg
csavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_m_read_readvariableopV
Rsavev2_adam_transformer_decoder_23_layer_normalization_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_23_layer_normalization_beta_m_read_readvariableopX
Tsavev2_adam_transformer_decoder_23_layer_normalization_1_gamma_m_read_readvariableopW
Ssavev2_adam_transformer_decoder_23_layer_normalization_1_beta_m_read_readvariableopI
Esavev2_adam_transformer_decoder_23_dense_kernel_m_read_readvariableopG
Csavev2_adam_transformer_decoder_23_dense_bias_m_read_readvariableopK
Gsavev2_adam_transformer_decoder_23_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_transformer_decoder_23_dense_1_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableopb
^savev2_adam_token_and_position_embedding_27_token_embedding28_embeddings_v_read_readvariableope
asavev2_adam_token_and_position_embedding_27_position_embedding28_embeddings_v_read_readvariableopb
^savev2_adam_token_and_position_embedding_28_token_embedding29_embeddings_v_read_readvariableope
asavev2_adam_token_and_position_embedding_28_position_embedding29_embeddings_v_read_readvariableop^
Zsavev2_adam_transformer_encoder_23_multi_head_attention_query_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_encoder_23_multi_head_attention_query_bias_v_read_readvariableop\
Xsavev2_adam_transformer_encoder_23_multi_head_attention_key_kernel_v_read_readvariableopZ
Vsavev2_adam_transformer_encoder_23_multi_head_attention_key_bias_v_read_readvariableop^
Zsavev2_adam_transformer_encoder_23_multi_head_attention_value_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_encoder_23_multi_head_attention_value_bias_v_read_readvariableopi
esavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_v_read_readvariableopg
csavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_v_read_readvariableopV
Rsavev2_adam_transformer_encoder_23_layer_normalization_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_encoder_23_layer_normalization_beta_v_read_readvariableopX
Tsavev2_adam_transformer_encoder_23_layer_normalization_1_gamma_v_read_readvariableopW
Ssavev2_adam_transformer_encoder_23_layer_normalization_1_beta_v_read_readvariableopI
Esavev2_adam_transformer_encoder_23_dense_kernel_v_read_readvariableopG
Csavev2_adam_transformer_encoder_23_dense_bias_v_read_readvariableopK
Gsavev2_adam_transformer_encoder_23_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_transformer_encoder_23_dense_1_bias_v_read_readvariableop^
Zsavev2_adam_transformer_decoder_23_multi_head_attention_query_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_decoder_23_multi_head_attention_query_bias_v_read_readvariableop\
Xsavev2_adam_transformer_decoder_23_multi_head_attention_key_kernel_v_read_readvariableopZ
Vsavev2_adam_transformer_decoder_23_multi_head_attention_key_bias_v_read_readvariableop^
Zsavev2_adam_transformer_decoder_23_multi_head_attention_value_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_decoder_23_multi_head_attention_value_bias_v_read_readvariableopi
esavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_v_read_readvariableopg
csavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_v_read_readvariableopV
Rsavev2_adam_transformer_decoder_23_layer_normalization_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_23_layer_normalization_beta_v_read_readvariableopX
Tsavev2_adam_transformer_decoder_23_layer_normalization_1_gamma_v_read_readvariableopW
Ssavev2_adam_transformer_decoder_23_layer_normalization_1_beta_v_read_readvariableopI
Esavev2_adam_transformer_decoder_23_dense_kernel_v_read_readvariableopG
Csavev2_adam_transformer_decoder_23_dense_bias_v_read_readvariableopK
Gsavev2_adam_transformer_decoder_23_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_transformer_decoder_23_dense_1_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?>
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?>
value?=B?=?B6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?R
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableopWsavev2_token_and_position_embedding_27_token_embedding28_embeddings_read_readvariableopZsavev2_token_and_position_embedding_27_position_embedding28_embeddings_read_readvariableopWsavev2_token_and_position_embedding_28_token_embedding29_embeddings_read_readvariableopZsavev2_token_and_position_embedding_28_position_embedding29_embeddings_read_readvariableopSsavev2_transformer_encoder_23_multi_head_attention_query_kernel_read_readvariableopQsavev2_transformer_encoder_23_multi_head_attention_query_bias_read_readvariableopQsavev2_transformer_encoder_23_multi_head_attention_key_kernel_read_readvariableopOsavev2_transformer_encoder_23_multi_head_attention_key_bias_read_readvariableopSsavev2_transformer_encoder_23_multi_head_attention_value_kernel_read_readvariableopQsavev2_transformer_encoder_23_multi_head_attention_value_bias_read_readvariableop^savev2_transformer_encoder_23_multi_head_attention_attention_output_kernel_read_readvariableop\savev2_transformer_encoder_23_multi_head_attention_attention_output_bias_read_readvariableopKsavev2_transformer_encoder_23_layer_normalization_gamma_read_readvariableopJsavev2_transformer_encoder_23_layer_normalization_beta_read_readvariableopMsavev2_transformer_encoder_23_layer_normalization_1_gamma_read_readvariableopLsavev2_transformer_encoder_23_layer_normalization_1_beta_read_readvariableop>savev2_transformer_encoder_23_dense_kernel_read_readvariableop<savev2_transformer_encoder_23_dense_bias_read_readvariableop@savev2_transformer_encoder_23_dense_1_kernel_read_readvariableop>savev2_transformer_encoder_23_dense_1_bias_read_readvariableopSsavev2_transformer_decoder_23_multi_head_attention_query_kernel_read_readvariableopQsavev2_transformer_decoder_23_multi_head_attention_query_bias_read_readvariableopQsavev2_transformer_decoder_23_multi_head_attention_key_kernel_read_readvariableopOsavev2_transformer_decoder_23_multi_head_attention_key_bias_read_readvariableopSsavev2_transformer_decoder_23_multi_head_attention_value_kernel_read_readvariableopQsavev2_transformer_decoder_23_multi_head_attention_value_bias_read_readvariableop^savev2_transformer_decoder_23_multi_head_attention_attention_output_kernel_read_readvariableop\savev2_transformer_decoder_23_multi_head_attention_attention_output_bias_read_readvariableopKsavev2_transformer_decoder_23_layer_normalization_gamma_read_readvariableopJsavev2_transformer_decoder_23_layer_normalization_beta_read_readvariableopMsavev2_transformer_decoder_23_layer_normalization_1_gamma_read_readvariableopLsavev2_transformer_decoder_23_layer_normalization_1_beta_read_readvariableop>savev2_transformer_decoder_23_dense_kernel_read_readvariableop<savev2_transformer_decoder_23_dense_bias_read_readvariableop@savev2_transformer_decoder_23_dense_1_kernel_read_readvariableop>savev2_transformer_decoder_23_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop^savev2_adam_token_and_position_embedding_27_token_embedding28_embeddings_m_read_readvariableopasavev2_adam_token_and_position_embedding_27_position_embedding28_embeddings_m_read_readvariableop^savev2_adam_token_and_position_embedding_28_token_embedding29_embeddings_m_read_readvariableopasavev2_adam_token_and_position_embedding_28_position_embedding29_embeddings_m_read_readvariableopZsavev2_adam_transformer_encoder_23_multi_head_attention_query_kernel_m_read_readvariableopXsavev2_adam_transformer_encoder_23_multi_head_attention_query_bias_m_read_readvariableopXsavev2_adam_transformer_encoder_23_multi_head_attention_key_kernel_m_read_readvariableopVsavev2_adam_transformer_encoder_23_multi_head_attention_key_bias_m_read_readvariableopZsavev2_adam_transformer_encoder_23_multi_head_attention_value_kernel_m_read_readvariableopXsavev2_adam_transformer_encoder_23_multi_head_attention_value_bias_m_read_readvariableopesavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_m_read_readvariableopcsavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_m_read_readvariableopRsavev2_adam_transformer_encoder_23_layer_normalization_gamma_m_read_readvariableopQsavev2_adam_transformer_encoder_23_layer_normalization_beta_m_read_readvariableopTsavev2_adam_transformer_encoder_23_layer_normalization_1_gamma_m_read_readvariableopSsavev2_adam_transformer_encoder_23_layer_normalization_1_beta_m_read_readvariableopEsavev2_adam_transformer_encoder_23_dense_kernel_m_read_readvariableopCsavev2_adam_transformer_encoder_23_dense_bias_m_read_readvariableopGsavev2_adam_transformer_encoder_23_dense_1_kernel_m_read_readvariableopEsavev2_adam_transformer_encoder_23_dense_1_bias_m_read_readvariableopZsavev2_adam_transformer_decoder_23_multi_head_attention_query_kernel_m_read_readvariableopXsavev2_adam_transformer_decoder_23_multi_head_attention_query_bias_m_read_readvariableopXsavev2_adam_transformer_decoder_23_multi_head_attention_key_kernel_m_read_readvariableopVsavev2_adam_transformer_decoder_23_multi_head_attention_key_bias_m_read_readvariableopZsavev2_adam_transformer_decoder_23_multi_head_attention_value_kernel_m_read_readvariableopXsavev2_adam_transformer_decoder_23_multi_head_attention_value_bias_m_read_readvariableopesavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_m_read_readvariableopcsavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_m_read_readvariableopRsavev2_adam_transformer_decoder_23_layer_normalization_gamma_m_read_readvariableopQsavev2_adam_transformer_decoder_23_layer_normalization_beta_m_read_readvariableopTsavev2_adam_transformer_decoder_23_layer_normalization_1_gamma_m_read_readvariableopSsavev2_adam_transformer_decoder_23_layer_normalization_1_beta_m_read_readvariableopEsavev2_adam_transformer_decoder_23_dense_kernel_m_read_readvariableopCsavev2_adam_transformer_decoder_23_dense_bias_m_read_readvariableopGsavev2_adam_transformer_decoder_23_dense_1_kernel_m_read_readvariableopEsavev2_adam_transformer_decoder_23_dense_1_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop^savev2_adam_token_and_position_embedding_27_token_embedding28_embeddings_v_read_readvariableopasavev2_adam_token_and_position_embedding_27_position_embedding28_embeddings_v_read_readvariableop^savev2_adam_token_and_position_embedding_28_token_embedding29_embeddings_v_read_readvariableopasavev2_adam_token_and_position_embedding_28_position_embedding29_embeddings_v_read_readvariableopZsavev2_adam_transformer_encoder_23_multi_head_attention_query_kernel_v_read_readvariableopXsavev2_adam_transformer_encoder_23_multi_head_attention_query_bias_v_read_readvariableopXsavev2_adam_transformer_encoder_23_multi_head_attention_key_kernel_v_read_readvariableopVsavev2_adam_transformer_encoder_23_multi_head_attention_key_bias_v_read_readvariableopZsavev2_adam_transformer_encoder_23_multi_head_attention_value_kernel_v_read_readvariableopXsavev2_adam_transformer_encoder_23_multi_head_attention_value_bias_v_read_readvariableopesavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_v_read_readvariableopcsavev2_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_v_read_readvariableopRsavev2_adam_transformer_encoder_23_layer_normalization_gamma_v_read_readvariableopQsavev2_adam_transformer_encoder_23_layer_normalization_beta_v_read_readvariableopTsavev2_adam_transformer_encoder_23_layer_normalization_1_gamma_v_read_readvariableopSsavev2_adam_transformer_encoder_23_layer_normalization_1_beta_v_read_readvariableopEsavev2_adam_transformer_encoder_23_dense_kernel_v_read_readvariableopCsavev2_adam_transformer_encoder_23_dense_bias_v_read_readvariableopGsavev2_adam_transformer_encoder_23_dense_1_kernel_v_read_readvariableopEsavev2_adam_transformer_encoder_23_dense_1_bias_v_read_readvariableopZsavev2_adam_transformer_decoder_23_multi_head_attention_query_kernel_v_read_readvariableopXsavev2_adam_transformer_decoder_23_multi_head_attention_query_bias_v_read_readvariableopXsavev2_adam_transformer_decoder_23_multi_head_attention_key_kernel_v_read_readvariableopVsavev2_adam_transformer_decoder_23_multi_head_attention_key_bias_v_read_readvariableopZsavev2_adam_transformer_decoder_23_multi_head_attention_value_kernel_v_read_readvariableopXsavev2_adam_transformer_decoder_23_multi_head_attention_value_bias_v_read_readvariableopesavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_v_read_readvariableopcsavev2_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_v_read_readvariableopRsavev2_adam_transformer_decoder_23_layer_normalization_gamma_v_read_readvariableopQsavev2_adam_transformer_decoder_23_layer_normalization_beta_v_read_readvariableopTsavev2_adam_transformer_decoder_23_layer_normalization_1_gamma_v_read_readvariableopSsavev2_adam_transformer_decoder_23_layer_normalization_1_beta_v_read_readvariableopEsavev2_adam_transformer_decoder_23_dense_kernel_v_read_readvariableopCsavev2_adam_transformer_decoder_23_dense_bias_v_read_readvariableopGsavev2_adam_transformer_decoder_23_dense_1_kernel_v_read_readvariableopEsavev2_adam_transformer_decoder_23_dense_1_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?	
_input_shapes?
?: :@:@:@::	?: :: ::::::::::::: : : :::::::::::::: : : :: : : : : ::: : : : :@:@:@::	?: :: ::::::::::::: : : :::::::::::::: : : ::@:@:@::	?: :: ::::::::::::: : : :::::::::::::: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	?:$ 

_output_shapes

: :$ 

_output_shapes

::$ 

_output_shapes

: :(	$
"
_output_shapes
::$
 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
::$% 

_output_shapes

: : &

_output_shapes
: :$' 

_output_shapes

: : (

_output_shapes
::)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
::/

_output_shapes
::0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :$4 

_output_shapes

:@: 5

_output_shapes
:@:$6 

_output_shapes

:@: 7

_output_shapes
::%8!

_output_shapes
:	?:$9 

_output_shapes

: :$: 

_output_shapes

::$; 

_output_shapes

: :(<$
"
_output_shapes
::$= 

_output_shapes

::(>$
"
_output_shapes
::$? 

_output_shapes

::(@$
"
_output_shapes
::$A 

_output_shapes

::(B$
"
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::$H 

_output_shapes

: : I

_output_shapes
: :$J 

_output_shapes

: : K

_output_shapes
::(L$
"
_output_shapes
::$M 

_output_shapes

::(N$
"
_output_shapes
::$O 

_output_shapes

::(P$
"
_output_shapes
::$Q 

_output_shapes

::(R$
"
_output_shapes
:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::$X 

_output_shapes

: : Y

_output_shapes
: :$Z 

_output_shapes

: : [

_output_shapes
::$\ 

_output_shapes

:@: ]

_output_shapes
:@:$^ 

_output_shapes

:@: _

_output_shapes
::%`!

_output_shapes
:	?:$a 

_output_shapes

: :$b 

_output_shapes

::$c 

_output_shapes

: :(d$
"
_output_shapes
::$e 

_output_shapes

::(f$
"
_output_shapes
::$g 

_output_shapes

::(h$
"
_output_shapes
::$i 

_output_shapes

::(j$
"
_output_shapes
:: k

_output_shapes
:: l

_output_shapes
:: m

_output_shapes
:: n

_output_shapes
:: o

_output_shapes
::$p 

_output_shapes

: : q

_output_shapes
: :$r 

_output_shapes

: : s

_output_shapes
::(t$
"
_output_shapes
::$u 

_output_shapes

::(v$
"
_output_shapes
::$w 

_output_shapes

::(x$
"
_output_shapes
::$y 

_output_shapes

::(z$
"
_output_shapes
:: {

_output_shapes
:: |

_output_shapes
:: }

_output_shapes
:: ~

_output_shapes
:: 

_output_shapes
::%? 

_output_shapes

: :!?

_output_shapes
: :%? 

_output_shapes

: :!?

_output_shapes
::?

_output_shapes
: 
?
-
__inference__destroyer_766064
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
-
__inference__destroyer_766049
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?!
?
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_762051

inputs	<
)token_embedding28_embedding_lookup_762027:	?>
,position_embedding28_readvariableop_resource: 
identity??#position_embedding28/ReadVariableOp?"token_embedding28/embedding_lookup?
"token_embedding28/embedding_lookupResourceGather)token_embedding28_embedding_lookup_762027inputs*
Tindices0	*<
_class2
0.loc:@token_embedding28/embedding_lookup/762027*+
_output_shapes
:????????? *
dtype0?
+token_embedding28/embedding_lookup/IdentityIdentity+token_embedding28/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding28/embedding_lookup/762027*+
_output_shapes
:????????? ?
-token_embedding28/embedding_lookup/Identity_1Identity4token_embedding28/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
position_embedding28/ShapeShape6token_embedding28/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding28/strided_sliceStridedSlice#position_embedding28/Shape:output:01position_embedding28/strided_slice/stack:output:03position_embedding28/strided_slice/stack_1:output:03position_embedding28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding28/ReadVariableOpReadVariableOp,position_embedding28_readvariableop_resource*
_output_shapes

: *
dtype0\
position_embedding28/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding28/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding28/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding28/strided_slice_1/stackPack#position_embedding28/Const:output:05position_embedding28/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding28/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding28/strided_slice_1/stack_1Pack+position_embedding28/strided_slice:output:07position_embedding28/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding28/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding28/strided_slice_1/stack_2Pack%position_embedding28/Const_1:output:07position_embedding28/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding28/strided_slice_1StridedSlice+position_embedding28/ReadVariableOp:value:03position_embedding28/strided_slice_1/stack:output:05position_embedding28/strided_slice_1/stack_1:output:05position_embedding28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
 position_embedding28/BroadcastToBroadcastTo-position_embedding28/strided_slice_1:output:0#position_embedding28/Shape:output:0*
T0*+
_output_shapes
:????????? ?
addAddV26token_embedding28/embedding_lookup/Identity_1:output:0)position_embedding28/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp$^position_embedding28/ReadVariableOp#^token_embedding28/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding28/ReadVariableOp#position_embedding28/ReadVariableOp2H
"token_embedding28/embedding_lookup"token_embedding28/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?5
D__inference_model_12_layer_call_and_return_conditional_losses_764619
inputs_0
inputs_1U
Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_13_string_lookup_13_equal_y5
1text_vectorization_13_string_lookup_13_selectv2_t	\
Itoken_and_position_embedding_27_token_embedding28_embedding_lookup_764261:	?^
Ltoken_and_position_embedding_27_position_embedding28_readvariableop_resource: [
Itoken_and_position_embedding_28_token_embedding29_embedding_lookup_764285:^
Ltoken_and_position_embedding_28_position_embedding29_readvariableop_resource: m
Wtransformer_encoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource:_
Mtransformer_encoder_23_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_encoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource:]
Ktransformer_encoder_23_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_encoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource:_
Mtransformer_encoder_23_multi_head_attention_value_add_readvariableop_resource:x
btransformer_encoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:f
Xtransformer_encoder_23_multi_head_attention_attention_output_add_readvariableop_resource:^
Ptransformer_encoder_23_layer_normalization_batchnorm_mul_readvariableop_resource:Z
Ltransformer_encoder_23_layer_normalization_batchnorm_readvariableop_resource:P
>transformer_encoder_23_dense_tensordot_readvariableop_resource: J
<transformer_encoder_23_dense_biasadd_readvariableop_resource: R
@transformer_encoder_23_dense_1_tensordot_readvariableop_resource: L
>transformer_encoder_23_dense_1_biasadd_readvariableop_resource:`
Rtransformer_encoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource:\
Ntransformer_encoder_23_layer_normalization_1_batchnorm_readvariableop_resource:m
Wtransformer_decoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource:_
Mtransformer_decoder_23_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_decoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource:]
Ktransformer_decoder_23_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_decoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource:_
Mtransformer_decoder_23_multi_head_attention_value_add_readvariableop_resource:x
btransformer_decoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:f
Xtransformer_decoder_23_multi_head_attention_attention_output_add_readvariableop_resource:^
Ptransformer_decoder_23_layer_normalization_batchnorm_mul_readvariableop_resource:Z
Ltransformer_decoder_23_layer_normalization_batchnorm_readvariableop_resource:P
>transformer_decoder_23_dense_tensordot_readvariableop_resource: J
<transformer_decoder_23_dense_biasadd_readvariableop_resource: R
@transformer_decoder_23_dense_1_tensordot_readvariableop_resource: L
>transformer_decoder_23_dense_1_biasadd_readvariableop_resource:`
Rtransformer_decoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource:\
Ntransformer_decoder_23_layer_normalization_1_batchnorm_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:@9
'dense_24_matmul_readvariableop_resource:@6
(dense_24_biasadd_readvariableop_resource:
identity??dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?Ctoken_and_position_embedding_27/position_embedding28/ReadVariableOp?Btoken_and_position_embedding_27/token_embedding28/embedding_lookup?Ctoken_and_position_embedding_28/position_embedding29/ReadVariableOp?Btoken_and_position_embedding_28/token_embedding29/embedding_lookup?3transformer_decoder_23/dense/BiasAdd/ReadVariableOp?5transformer_decoder_23/dense/Tensordot/ReadVariableOp?5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp?7transformer_decoder_23/dense_1/Tensordot/ReadVariableOp?Ctransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_decoder_23/multi_head_attention/key/add/ReadVariableOp?Ltransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOp?Ntransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOp?Ntransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp?3transformer_encoder_23/dense/BiasAdd/ReadVariableOp?5transformer_encoder_23/dense/Tensordot/ReadVariableOp?5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp?7transformer_encoder_23/dense_1/Tensordot/ReadVariableOp?Ctransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_encoder_23/multi_head_attention/key/add/ReadVariableOp?Ltransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOp?Ntransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOp?Ntransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
text_vectorization_13/SqueezeSqueezeinputs_0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????h
'text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2&text_vectorization_13/Squeeze:output:00text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
5text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/text_vectorization_13/StringSplit/strided_sliceStridedSlice9text_vectorization_13/StringSplit/StringSplitV2:indices:0>text_vectorization_13/StringSplit/strided_slice/stack:output:0@text_vectorization_13/StringSplit/strided_slice/stack_1:output:0@text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
7text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1text_vectorization_13/StringSplit/strided_slice_1StridedSlice7text_vectorization_13/StringSplit/StringSplitV2:shape:0@text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Xtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ftext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0stext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
etext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountmtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ctext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle8text_vectorization_13/StringSplit/StringSplitV2:values:0Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_13/string_lookup_13/EqualEqual8text_vectorization_13/StringSplit/StringSplitV2:values:0.text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/SelectV2SelectV20text_vectorization_13/string_lookup_13/Equal:z:01text_vectorization_13_string_lookup_13_selectv2_tMtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/IdentityIdentity8text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????t
2text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
*text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
9text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_13/RaggedToTensor/Const:output:08text_vectorization_13/string_lookup_13/Identity:output:0;text_vectorization_13/RaggedToTensor/default_value:output:0:text_vectorization_13/StringSplit/strided_slice_1:output:08text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Btoken_and_position_embedding_27/token_embedding28/embedding_lookupResourceGatherItoken_and_position_embedding_27_token_embedding28_embedding_lookup_764261Btext_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*\
_classR
PNloc:@token_and_position_embedding_27/token_embedding28/embedding_lookup/764261*+
_output_shapes
:????????? *
dtype0?
Ktoken_and_position_embedding_27/token_embedding28/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_27/token_embedding28/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_27/token_embedding28/embedding_lookup/764261*+
_output_shapes
:????????? ?
Mtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
:token_and_position_embedding_27/position_embedding28/ShapeShapeVtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_27/position_embedding28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_27/position_embedding28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_27/position_embedding28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_27/position_embedding28/strided_sliceStridedSliceCtoken_and_position_embedding_27/position_embedding28/Shape:output:0Qtoken_and_position_embedding_27/position_embedding28/strided_slice/stack:output:0Stoken_and_position_embedding_27/position_embedding28/strided_slice/stack_1:output:0Stoken_and_position_embedding_27/position_embedding28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_27/position_embedding28/ReadVariableOpReadVariableOpLtoken_and_position_embedding_27_position_embedding28_readvariableop_resource*
_output_shapes

: *
dtype0|
:token_and_position_embedding_27/position_embedding28/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_27/position_embedding28/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_27/position_embedding28/strided_slice_1/stackPackCtoken_and_position_embedding_27/position_embedding28/Const:output:0Utoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1PackKtoken_and_position_embedding_27/position_embedding28/strided_slice:output:0Wtoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2PackEtoken_and_position_embedding_27/position_embedding28/Const_1:output:0Wtoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_27/position_embedding28/strided_slice_1StridedSliceKtoken_and_position_embedding_27/position_embedding28/ReadVariableOp:value:0Stoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack:output:0Utoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
@token_and_position_embedding_27/position_embedding28/BroadcastToBroadcastToMtoken_and_position_embedding_27/position_embedding28/strided_slice_1:output:0Ctoken_and_position_embedding_27/position_embedding28/Shape:output:0*
T0*+
_output_shapes
:????????? ?
#token_and_position_embedding_27/addAddV2Vtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_27/position_embedding28/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? ?
6token_and_position_embedding_28/token_embedding29/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
Btoken_and_position_embedding_28/token_embedding29/embedding_lookupResourceGatherItoken_and_position_embedding_28_token_embedding29_embedding_lookup_764285:token_and_position_embedding_28/token_embedding29/Cast:y:0*
Tindices0*\
_classR
PNloc:@token_and_position_embedding_28/token_embedding29/embedding_lookup/764285*+
_output_shapes
:????????? *
dtype0?
Ktoken_and_position_embedding_28/token_embedding29/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_28/token_embedding29/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_28/token_embedding29/embedding_lookup/764285*+
_output_shapes
:????????? ?
Mtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
:token_and_position_embedding_28/position_embedding29/ShapeShapeVtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_28/position_embedding29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_28/position_embedding29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_28/position_embedding29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_28/position_embedding29/strided_sliceStridedSliceCtoken_and_position_embedding_28/position_embedding29/Shape:output:0Qtoken_and_position_embedding_28/position_embedding29/strided_slice/stack:output:0Stoken_and_position_embedding_28/position_embedding29/strided_slice/stack_1:output:0Stoken_and_position_embedding_28/position_embedding29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_28/position_embedding29/ReadVariableOpReadVariableOpLtoken_and_position_embedding_28_position_embedding29_readvariableop_resource*
_output_shapes

: *
dtype0|
:token_and_position_embedding_28/position_embedding29/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_28/position_embedding29/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_28/position_embedding29/strided_slice_1/stackPackCtoken_and_position_embedding_28/position_embedding29/Const:output:0Utoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1PackKtoken_and_position_embedding_28/position_embedding29/strided_slice:output:0Wtoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2PackEtoken_and_position_embedding_28/position_embedding29/Const_1:output:0Wtoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_28/position_embedding29/strided_slice_1StridedSliceKtoken_and_position_embedding_28/position_embedding29/ReadVariableOp:value:0Stoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack:output:0Utoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
@token_and_position_embedding_28/position_embedding29/BroadcastToBroadcastToMtoken_and_position_embedding_28/position_embedding29/strided_slice_1:output:0Ctoken_and_position_embedding_28/position_embedding29/Shape:output:0*
T0*+
_output_shapes
:????????? ?
#token_and_position_embedding_28/addAddV2Vtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_28/position_embedding29/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? ?

add_12/addAddV2'token_and_position_embedding_27/add:z:0'token_and_position_embedding_28/add:z:0*
T0*+
_output_shapes
:????????? ?
Ntransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_encoder_23/multi_head_attention/query/einsum/EinsumEinsumadd_12/add:z:0Vtransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_encoder_23_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_23/multi_head_attention/query/addAddV2Htransformer_encoder_23/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_encoder_23/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_encoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
=transformer_encoder_23/multi_head_attention/key/einsum/EinsumEinsumadd_12/add:z:0Ttransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_encoder_23/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_encoder_23_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_encoder_23/multi_head_attention/key/addAddV2Ftransformer_encoder_23/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_encoder_23/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_encoder_23/multi_head_attention/value/einsum/EinsumEinsumadd_12/add:z:0Vtransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_encoder_23_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_23/multi_head_attention/value/addAddV2Htransformer_encoder_23/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_encoder_23/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_encoder_23/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
/transformer_encoder_23/multi_head_attention/MulMul9transformer_encoder_23/multi_head_attention/query/add:z:0:transformer_encoder_23/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_encoder_23/multi_head_attention/einsum/EinsumEinsum7transformer_encoder_23/multi_head_attention/key/add:z:03transformer_encoder_23/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
;transformer_encoder_23/multi_head_attention/softmax/SoftmaxSoftmaxBtransformer_encoder_23/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
>transformer_encoder_23/multi_head_attention/dropout_2/IdentityIdentityEtransformer_encoder_23/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
;transformer_encoder_23/multi_head_attention/einsum_1/EinsumEinsumGtransformer_encoder_23/multi_head_attention/dropout_2/Identity:output:09transformer_encoder_23/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_encoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Jtransformer_encoder_23/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_encoder_23/multi_head_attention/einsum_1/Einsum:output:0atransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Otransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_encoder_23_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
@transformer_encoder_23/multi_head_attention/attention_output/addAddV2Stransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
'transformer_encoder_23/dropout/IdentityIdentityDtransformer_encoder_23/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_23/addAddV2add_12/add:z:00transformer_encoder_23/dropout/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Itransformer_encoder_23/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_encoder_23/layer_normalization/moments/meanMeantransformer_encoder_23/add:z:0Rtransformer_encoder_23/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_encoder_23/layer_normalization/moments/StopGradientStopGradient@transformer_encoder_23/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_encoder_23/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_23/add:z:0Htransformer_encoder_23/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Mtransformer_encoder_23/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_encoder_23/layer_normalization/moments/varianceMeanHtransformer_encoder_23/layer_normalization/moments/SquaredDifference:z:0Vtransformer_encoder_23/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_encoder_23/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_encoder_23/layer_normalization/batchnorm/addAddV2Dtransformer_encoder_23/layer_normalization/moments/variance:output:0Ctransformer_encoder_23/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/RsqrtRsqrt<transformer_encoder_23/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_encoder_23_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_encoder_23/layer_normalization/batchnorm/mulMul>transformer_encoder_23/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/mul_1Multransformer_encoder_23/add:z:0<transformer_encoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/mul_2Mul@transformer_encoder_23/layer_normalization/moments/mean:output:0<transformer_encoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_encoder_23_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_encoder_23/layer_normalization/batchnorm/subSubKtransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_encoder_23/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/add_1AddV2>transformer_encoder_23/layer_normalization/batchnorm/mul_1:z:0<transformer_encoder_23/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
5transformer_encoder_23/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_encoder_23_dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0u
+transformer_encoder_23/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_encoder_23/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_encoder_23/dense/Tensordot/ShapeShape>transformer_encoder_23/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_encoder_23/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_23/dense/Tensordot/GatherV2GatherV25transformer_encoder_23/dense/Tensordot/Shape:output:04transformer_encoder_23/dense/Tensordot/free:output:0=transformer_encoder_23/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_encoder_23/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_23/dense/Tensordot/GatherV2_1GatherV25transformer_encoder_23/dense/Tensordot/Shape:output:04transformer_encoder_23/dense/Tensordot/axes:output:0?transformer_encoder_23/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_encoder_23/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_encoder_23/dense/Tensordot/ProdProd8transformer_encoder_23/dense/Tensordot/GatherV2:output:05transformer_encoder_23/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_encoder_23/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_23/dense/Tensordot/Prod_1Prod:transformer_encoder_23/dense/Tensordot/GatherV2_1:output:07transformer_encoder_23/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_encoder_23/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_encoder_23/dense/Tensordot/concatConcatV24transformer_encoder_23/dense/Tensordot/free:output:04transformer_encoder_23/dense/Tensordot/axes:output:0;transformer_encoder_23/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_encoder_23/dense/Tensordot/stackPack4transformer_encoder_23/dense/Tensordot/Prod:output:06transformer_encoder_23/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_encoder_23/dense/Tensordot/transpose	Transpose>transformer_encoder_23/layer_normalization/batchnorm/add_1:z:06transformer_encoder_23/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
.transformer_encoder_23/dense/Tensordot/ReshapeReshape4transformer_encoder_23/dense/Tensordot/transpose:y:05transformer_encoder_23/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_encoder_23/dense/Tensordot/MatMulMatMul7transformer_encoder_23/dense/Tensordot/Reshape:output:0=transformer_encoder_23/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? x
.transformer_encoder_23/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: v
4transformer_encoder_23/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_23/dense/Tensordot/concat_1ConcatV28transformer_encoder_23/dense/Tensordot/GatherV2:output:07transformer_encoder_23/dense/Tensordot/Const_2:output:0=transformer_encoder_23/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_encoder_23/dense/TensordotReshape7transformer_encoder_23/dense/Tensordot/MatMul:product:08transformer_encoder_23/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
3transformer_encoder_23/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_23_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
$transformer_encoder_23/dense/BiasAddBiasAdd/transformer_encoder_23/dense/Tensordot:output:0;transformer_encoder_23/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
!transformer_encoder_23/dense/ReluRelu-transformer_encoder_23/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
7transformer_encoder_23/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_encoder_23_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0w
-transformer_encoder_23/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_encoder_23/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_encoder_23/dense_1/Tensordot/ShapeShape/transformer_encoder_23/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_encoder_23/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_23/dense_1/Tensordot/GatherV2GatherV27transformer_encoder_23/dense_1/Tensordot/Shape:output:06transformer_encoder_23/dense_1/Tensordot/free:output:0?transformer_encoder_23/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_encoder_23/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_encoder_23/dense_1/Tensordot/GatherV2_1GatherV27transformer_encoder_23/dense_1/Tensordot/Shape:output:06transformer_encoder_23/dense_1/Tensordot/axes:output:0Atransformer_encoder_23/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_encoder_23/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_23/dense_1/Tensordot/ProdProd:transformer_encoder_23/dense_1/Tensordot/GatherV2:output:07transformer_encoder_23/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_encoder_23/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_encoder_23/dense_1/Tensordot/Prod_1Prod<transformer_encoder_23/dense_1/Tensordot/GatherV2_1:output:09transformer_encoder_23/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_encoder_23/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_23/dense_1/Tensordot/concatConcatV26transformer_encoder_23/dense_1/Tensordot/free:output:06transformer_encoder_23/dense_1/Tensordot/axes:output:0=transformer_encoder_23/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_encoder_23/dense_1/Tensordot/stackPack6transformer_encoder_23/dense_1/Tensordot/Prod:output:08transformer_encoder_23/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_encoder_23/dense_1/Tensordot/transpose	Transpose/transformer_encoder_23/dense/Relu:activations:08transformer_encoder_23/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
0transformer_encoder_23/dense_1/Tensordot/ReshapeReshape6transformer_encoder_23/dense_1/Tensordot/transpose:y:07transformer_encoder_23/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_encoder_23/dense_1/Tensordot/MatMulMatMul9transformer_encoder_23/dense_1/Tensordot/Reshape:output:0?transformer_encoder_23/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
0transformer_encoder_23/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:x
6transformer_encoder_23/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_23/dense_1/Tensordot/concat_1ConcatV2:transformer_encoder_23/dense_1/Tensordot/GatherV2:output:09transformer_encoder_23/dense_1/Tensordot/Const_2:output:0?transformer_encoder_23/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_encoder_23/dense_1/TensordotReshape9transformer_encoder_23/dense_1/Tensordot/MatMul:product:0:transformer_encoder_23/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_encoder_23_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&transformer_encoder_23/dense_1/BiasAddBiasAdd1transformer_encoder_23/dense_1/Tensordot:output:0=transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
)transformer_encoder_23/dropout_1/IdentityIdentity/transformer_encoder_23/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_23/add_1AddV2>transformer_encoder_23/layer_normalization/batchnorm/add_1:z:02transformer_encoder_23/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Ktransformer_encoder_23/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_encoder_23/layer_normalization_1/moments/meanMean transformer_encoder_23/add_1:z:0Ttransformer_encoder_23/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_encoder_23/layer_normalization_1/moments/StopGradientStopGradientBtransformer_encoder_23/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_encoder_23/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_encoder_23/add_1:z:0Jtransformer_encoder_23/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Otransformer_encoder_23/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_encoder_23/layer_normalization_1/moments/varianceMeanJtransformer_encoder_23/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_encoder_23/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_encoder_23/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_encoder_23/layer_normalization_1/batchnorm/addAddV2Ftransformer_encoder_23/layer_normalization_1/moments/variance:output:0Etransformer_encoder_23/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_encoder_23/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_encoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_encoder_23/layer_normalization_1/batchnorm/mulMul@transformer_encoder_23/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/mul_1Mul transformer_encoder_23/add_1:z:0>transformer_encoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/mul_2MulBtransformer_encoder_23/layer_normalization_1/moments/mean:output:0>transformer_encoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Etransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_encoder_23_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_encoder_23/layer_normalization_1/batchnorm/subSubMtransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_encoder_23/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/add_1AddV2@transformer_encoder_23/layer_normalization_1/batchnorm/mul_1:z:0>transformer_encoder_23/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_23/ShapeShape@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:t
*transformer_decoder_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,transformer_decoder_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,transformer_decoder_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$transformer_decoder_23/strided_sliceStridedSlice%transformer_decoder_23/Shape:output:03transformer_decoder_23/strided_slice/stack:output:05transformer_decoder_23/strided_slice/stack_1:output:05transformer_decoder_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_23/strided_slice_1StridedSlice%transformer_decoder_23/Shape:output:05transformer_decoder_23/strided_slice_1/stack:output:07transformer_decoder_23/strided_slice_1/stack_1:output:07transformer_decoder_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"transformer_decoder_23/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"transformer_decoder_23/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_23/rangeRange+transformer_decoder_23/range/start:output:0/transformer_decoder_23/strided_slice_1:output:0+transformer_decoder_23/range/delta:output:0*
_output_shapes
: }
,transformer_decoder_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&transformer_decoder_23/strided_slice_2StridedSlice%transformer_decoder_23/range:output:05transformer_decoder_23/strided_slice_2/stack:output:07transformer_decoder_23/strided_slice_2/stack_1:output:07transformer_decoder_23/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskf
$transformer_decoder_23/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : f
$transformer_decoder_23/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_23/range_1Range-transformer_decoder_23/range_1/start:output:0/transformer_decoder_23/strided_slice_1:output:0-transformer_decoder_23/range_1/delta:output:0*
_output_shapes
: ?
#transformer_decoder_23/GreaterEqualGreaterEqual/transformer_decoder_23/strided_slice_2:output:0'transformer_decoder_23/range_1:output:0*
T0*
_output_shapes

:  ?
transformer_decoder_23/CastCast'transformer_decoder_23/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  v
,transformer_decoder_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_23/strided_slice_3StridedSlice%transformer_decoder_23/Shape:output:05transformer_decoder_23/strided_slice_3/stack:output:07transformer_decoder_23/strided_slice_3/stack_1:output:07transformer_decoder_23/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_23/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_23/strided_slice_4StridedSlice%transformer_decoder_23/Shape:output:05transformer_decoder_23/strided_slice_4/stack:output:07transformer_decoder_23/strided_slice_4/stack_1:output:07transformer_decoder_23/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&transformer_decoder_23/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
$transformer_decoder_23/Reshape/shapePack/transformer_decoder_23/Reshape/shape/0:output:0/transformer_decoder_23/strided_slice_3:output:0/transformer_decoder_23/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_23/ReshapeReshapetransformer_decoder_23/Cast:y:0-transformer_decoder_23/Reshape/shape:output:0*
T0*"
_output_shapes
:  p
%transformer_decoder_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!transformer_decoder_23/ExpandDims
ExpandDims-transformer_decoder_23/strided_slice:output:0.transformer_decoder_23/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
transformer_decoder_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"      d
"transformer_decoder_23/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
transformer_decoder_23/concatConcatV2*transformer_decoder_23/ExpandDims:output:0%transformer_decoder_23/Const:output:0+transformer_decoder_23/concat/axis:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_23/TileTile'transformer_decoder_23/Reshape:output:0&transformer_decoder_23/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Ntransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_decoder_23/multi_head_attention/query/einsum/EinsumEinsum@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_decoder_23_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_23/multi_head_attention/query/addAddV2Htransformer_decoder_23/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_decoder_23/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_decoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
=transformer_decoder_23/multi_head_attention/key/einsum/EinsumEinsum@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0Ttransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_decoder_23/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_decoder_23_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_decoder_23/multi_head_attention/key/addAddV2Ftransformer_decoder_23/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_decoder_23/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_decoder_23/multi_head_attention/value/einsum/EinsumEinsum@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_decoder_23_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_23/multi_head_attention/value/addAddV2Htransformer_decoder_23/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_decoder_23/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_decoder_23/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
/transformer_decoder_23/multi_head_attention/MulMul9transformer_decoder_23/multi_head_attention/query/add:z:0:transformer_decoder_23/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_decoder_23/multi_head_attention/einsum/EinsumEinsum7transformer_decoder_23/multi_head_attention/key/add:z:03transformer_decoder_23/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
:transformer_decoder_23/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6transformer_decoder_23/multi_head_attention/ExpandDims
ExpandDims$transformer_decoder_23/Tile:output:0Ctransformer_decoder_23/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
8transformer_decoder_23/multi_head_attention/softmax/CastCast?transformer_decoder_23/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ~
9transformer_decoder_23/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7transformer_decoder_23/multi_head_attention/softmax/subSubBtransformer_decoder_23/multi_head_attention/softmax/sub/x:output:0<transformer_decoder_23/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ~
9transformer_decoder_23/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
7transformer_decoder_23/multi_head_attention/softmax/mulMul;transformer_decoder_23/multi_head_attention/softmax/sub:z:0Btransformer_decoder_23/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
7transformer_decoder_23/multi_head_attention/softmax/addAddV2Btransformer_decoder_23/multi_head_attention/einsum/Einsum:output:0;transformer_decoder_23/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_23/multi_head_attention/softmax/SoftmaxSoftmax;transformer_decoder_23/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
>transformer_decoder_23/multi_head_attention/dropout_2/IdentityIdentityEtransformer_decoder_23/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_23/multi_head_attention/einsum_1/EinsumEinsumGtransformer_decoder_23/multi_head_attention/dropout_2/Identity:output:09transformer_decoder_23/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_decoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Jtransformer_decoder_23/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_decoder_23/multi_head_attention/einsum_1/Einsum:output:0atransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Otransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_decoder_23_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
@transformer_decoder_23/multi_head_attention/attention_output/addAddV2Stransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
'transformer_decoder_23/dropout/IdentityIdentityDtransformer_decoder_23/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_23/addAddV20transformer_decoder_23/dropout/Identity:output:0@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_decoder_23/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_decoder_23/layer_normalization/moments/meanMeantransformer_decoder_23/add:z:0Rtransformer_decoder_23/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_decoder_23/layer_normalization/moments/StopGradientStopGradient@transformer_decoder_23/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_decoder_23/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_23/add:z:0Htransformer_decoder_23/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Mtransformer_decoder_23/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_decoder_23/layer_normalization/moments/varianceMeanHtransformer_decoder_23/layer_normalization/moments/SquaredDifference:z:0Vtransformer_decoder_23/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_decoder_23/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_decoder_23/layer_normalization/batchnorm/addAddV2Dtransformer_decoder_23/layer_normalization/moments/variance:output:0Ctransformer_decoder_23/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/RsqrtRsqrt<transformer_decoder_23/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_decoder_23_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_decoder_23/layer_normalization/batchnorm/mulMul>transformer_decoder_23/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/mul_1Multransformer_decoder_23/add:z:0<transformer_decoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/mul_2Mul@transformer_decoder_23/layer_normalization/moments/mean:output:0<transformer_decoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_decoder_23_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_decoder_23/layer_normalization/batchnorm/subSubKtransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_decoder_23/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/add_1AddV2>transformer_decoder_23/layer_normalization/batchnorm/mul_1:z:0<transformer_decoder_23/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
5transformer_decoder_23/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_decoder_23_dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0u
+transformer_decoder_23/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_decoder_23/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_decoder_23/dense/Tensordot/ShapeShape>transformer_decoder_23/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_decoder_23/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_23/dense/Tensordot/GatherV2GatherV25transformer_decoder_23/dense/Tensordot/Shape:output:04transformer_decoder_23/dense/Tensordot/free:output:0=transformer_decoder_23/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_decoder_23/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_23/dense/Tensordot/GatherV2_1GatherV25transformer_decoder_23/dense/Tensordot/Shape:output:04transformer_decoder_23/dense/Tensordot/axes:output:0?transformer_decoder_23/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_decoder_23/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_decoder_23/dense/Tensordot/ProdProd8transformer_decoder_23/dense/Tensordot/GatherV2:output:05transformer_decoder_23/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_decoder_23/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_23/dense/Tensordot/Prod_1Prod:transformer_decoder_23/dense/Tensordot/GatherV2_1:output:07transformer_decoder_23/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_decoder_23/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_decoder_23/dense/Tensordot/concatConcatV24transformer_decoder_23/dense/Tensordot/free:output:04transformer_decoder_23/dense/Tensordot/axes:output:0;transformer_decoder_23/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_decoder_23/dense/Tensordot/stackPack4transformer_decoder_23/dense/Tensordot/Prod:output:06transformer_decoder_23/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_decoder_23/dense/Tensordot/transpose	Transpose>transformer_decoder_23/layer_normalization/batchnorm/add_1:z:06transformer_decoder_23/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
.transformer_decoder_23/dense/Tensordot/ReshapeReshape4transformer_decoder_23/dense/Tensordot/transpose:y:05transformer_decoder_23/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_decoder_23/dense/Tensordot/MatMulMatMul7transformer_decoder_23/dense/Tensordot/Reshape:output:0=transformer_decoder_23/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? x
.transformer_decoder_23/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: v
4transformer_decoder_23/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_23/dense/Tensordot/concat_1ConcatV28transformer_decoder_23/dense/Tensordot/GatherV2:output:07transformer_decoder_23/dense/Tensordot/Const_2:output:0=transformer_decoder_23/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_decoder_23/dense/TensordotReshape7transformer_decoder_23/dense/Tensordot/MatMul:product:08transformer_decoder_23/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
3transformer_decoder_23/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_decoder_23_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
$transformer_decoder_23/dense/BiasAddBiasAdd/transformer_decoder_23/dense/Tensordot:output:0;transformer_decoder_23/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
!transformer_decoder_23/dense/ReluRelu-transformer_decoder_23/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
7transformer_decoder_23/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_decoder_23_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0w
-transformer_decoder_23/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_decoder_23/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_decoder_23/dense_1/Tensordot/ShapeShape/transformer_decoder_23/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_decoder_23/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_23/dense_1/Tensordot/GatherV2GatherV27transformer_decoder_23/dense_1/Tensordot/Shape:output:06transformer_decoder_23/dense_1/Tensordot/free:output:0?transformer_decoder_23/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_decoder_23/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_decoder_23/dense_1/Tensordot/GatherV2_1GatherV27transformer_decoder_23/dense_1/Tensordot/Shape:output:06transformer_decoder_23/dense_1/Tensordot/axes:output:0Atransformer_decoder_23/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_decoder_23/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_23/dense_1/Tensordot/ProdProd:transformer_decoder_23/dense_1/Tensordot/GatherV2:output:07transformer_decoder_23/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_decoder_23/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_decoder_23/dense_1/Tensordot/Prod_1Prod<transformer_decoder_23/dense_1/Tensordot/GatherV2_1:output:09transformer_decoder_23/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_decoder_23/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_23/dense_1/Tensordot/concatConcatV26transformer_decoder_23/dense_1/Tensordot/free:output:06transformer_decoder_23/dense_1/Tensordot/axes:output:0=transformer_decoder_23/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_decoder_23/dense_1/Tensordot/stackPack6transformer_decoder_23/dense_1/Tensordot/Prod:output:08transformer_decoder_23/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_decoder_23/dense_1/Tensordot/transpose	Transpose/transformer_decoder_23/dense/Relu:activations:08transformer_decoder_23/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
0transformer_decoder_23/dense_1/Tensordot/ReshapeReshape6transformer_decoder_23/dense_1/Tensordot/transpose:y:07transformer_decoder_23/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_decoder_23/dense_1/Tensordot/MatMulMatMul9transformer_decoder_23/dense_1/Tensordot/Reshape:output:0?transformer_decoder_23/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
0transformer_decoder_23/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:x
6transformer_decoder_23/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_23/dense_1/Tensordot/concat_1ConcatV2:transformer_decoder_23/dense_1/Tensordot/GatherV2:output:09transformer_decoder_23/dense_1/Tensordot/Const_2:output:0?transformer_decoder_23/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_decoder_23/dense_1/TensordotReshape9transformer_decoder_23/dense_1/Tensordot/MatMul:product:0:transformer_decoder_23/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_decoder_23_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&transformer_decoder_23/dense_1/BiasAddBiasAdd1transformer_decoder_23/dense_1/Tensordot:output:0=transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
)transformer_decoder_23/dropout_1/IdentityIdentity/transformer_decoder_23/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_23/add_1AddV2>transformer_decoder_23/layer_normalization/batchnorm/add_1:z:02transformer_decoder_23/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Ktransformer_decoder_23/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_decoder_23/layer_normalization_1/moments/meanMean transformer_decoder_23/add_1:z:0Ttransformer_decoder_23/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_decoder_23/layer_normalization_1/moments/StopGradientStopGradientBtransformer_decoder_23/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_decoder_23/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_decoder_23/add_1:z:0Jtransformer_decoder_23/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Otransformer_decoder_23/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_decoder_23/layer_normalization_1/moments/varianceMeanJtransformer_decoder_23/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_decoder_23/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_decoder_23/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_decoder_23/layer_normalization_1/batchnorm/addAddV2Ftransformer_decoder_23/layer_normalization_1/moments/variance:output:0Etransformer_decoder_23/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_decoder_23/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_decoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_decoder_23/layer_normalization_1/batchnorm/mulMul@transformer_decoder_23/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/mul_1Mul transformer_decoder_23/add_1:z:0>transformer_decoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/mul_2MulBtransformer_decoder_23/layer_normalization_1/moments/mean:output:0>transformer_decoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Etransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_decoder_23_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_decoder_23/layer_normalization_1/batchnorm/subSubMtransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_decoder_23/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/add_1AddV2@transformer_decoder_23/layer_normalization_1/batchnorm/mul_1:z:0>transformer_decoder_23/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? t
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
 global_average_pooling1d_12/MeanMean@transformer_decoder_23/layer_normalization_1/batchnorm/add_1:z:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_23/MatMulMatMul)global_average_pooling1d_12/Mean:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@n
dropout_10/IdentityIdentitydense_23/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_24/MatMulMatMuldropout_10/Identity:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_24/SigmoidSigmoiddense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOpE^text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2D^token_and_position_embedding_27/position_embedding28/ReadVariableOpC^token_and_position_embedding_27/token_embedding28/embedding_lookupD^token_and_position_embedding_28/position_embedding29/ReadVariableOpC^token_and_position_embedding_28/token_embedding29/embedding_lookup4^transformer_decoder_23/dense/BiasAdd/ReadVariableOp6^transformer_decoder_23/dense/Tensordot/ReadVariableOp6^transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp8^transformer_decoder_23/dense_1/Tensordot/ReadVariableOpD^transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpH^transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_decoder_23/multi_head_attention/key/add/ReadVariableOpM^transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_decoder_23/multi_head_attention/query/add/ReadVariableOpO^transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_decoder_23/multi_head_attention/value/add/ReadVariableOpO^transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp4^transformer_encoder_23/dense/BiasAdd/ReadVariableOp6^transformer_encoder_23/dense/Tensordot/ReadVariableOp6^transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp8^transformer_encoder_23/dense_1/Tensordot/ReadVariableOpD^transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpH^transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_encoder_23/multi_head_attention/key/add/ReadVariableOpM^transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_encoder_23/multi_head_attention/query/add/ReadVariableOpO^transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_encoder_23/multi_head_attention/value/add/ReadVariableOpO^transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2?
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22?
Ctoken_and_position_embedding_27/position_embedding28/ReadVariableOpCtoken_and_position_embedding_27/position_embedding28/ReadVariableOp2?
Btoken_and_position_embedding_27/token_embedding28/embedding_lookupBtoken_and_position_embedding_27/token_embedding28/embedding_lookup2?
Ctoken_and_position_embedding_28/position_embedding29/ReadVariableOpCtoken_and_position_embedding_28/position_embedding29/ReadVariableOp2?
Btoken_and_position_embedding_28/token_embedding29/embedding_lookupBtoken_and_position_embedding_28/token_embedding29/embedding_lookup2j
3transformer_decoder_23/dense/BiasAdd/ReadVariableOp3transformer_decoder_23/dense/BiasAdd/ReadVariableOp2n
5transformer_decoder_23/dense/Tensordot/ReadVariableOp5transformer_decoder_23/dense/Tensordot/ReadVariableOp2n
5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp2r
7transformer_decoder_23/dense_1/Tensordot/ReadVariableOp7transformer_decoder_23/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpCtransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_decoder_23/multi_head_attention/key/add/ReadVariableOpBtransformer_decoder_23/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOpDtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOpDtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp2j
3transformer_encoder_23/dense/BiasAdd/ReadVariableOp3transformer_encoder_23/dense/BiasAdd/ReadVariableOp2n
5transformer_encoder_23/dense/Tensordot/ReadVariableOp5transformer_encoder_23/dense/Tensordot/ReadVariableOp2n
5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp2r
7transformer_encoder_23/dense_1/Tensordot/ReadVariableOp7transformer_encoder_23/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpCtransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_encoder_23/multi_head_attention/key/add/ReadVariableOpBtransformer_encoder_23/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOpDtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOpDtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?"
?
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_765151

inputs;
)token_embedding29_embedding_lookup_765127:>
,position_embedding29_readvariableop_resource: 
identity??#position_embedding29/ReadVariableOp?"token_embedding29/embedding_lookupg
token_embedding29/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
"token_embedding29/embedding_lookupResourceGather)token_embedding29_embedding_lookup_765127token_embedding29/Cast:y:0*
Tindices0*<
_class2
0.loc:@token_embedding29/embedding_lookup/765127*+
_output_shapes
:????????? *
dtype0?
+token_embedding29/embedding_lookup/IdentityIdentity+token_embedding29/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding29/embedding_lookup/765127*+
_output_shapes
:????????? ?
-token_embedding29/embedding_lookup/Identity_1Identity4token_embedding29/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
position_embedding29/ShapeShape6token_embedding29/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding29/strided_sliceStridedSlice#position_embedding29/Shape:output:01position_embedding29/strided_slice/stack:output:03position_embedding29/strided_slice/stack_1:output:03position_embedding29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding29/ReadVariableOpReadVariableOp,position_embedding29_readvariableop_resource*
_output_shapes

: *
dtype0\
position_embedding29/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding29/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding29/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding29/strided_slice_1/stackPack#position_embedding29/Const:output:05position_embedding29/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding29/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding29/strided_slice_1/stack_1Pack+position_embedding29/strided_slice:output:07position_embedding29/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding29/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding29/strided_slice_1/stack_2Pack%position_embedding29/Const_1:output:07position_embedding29/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding29/strided_slice_1StridedSlice+position_embedding29/ReadVariableOp:value:03position_embedding29/strided_slice_1/stack:output:05position_embedding29/strided_slice_1/stack_1:output:05position_embedding29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
 position_embedding29/BroadcastToBroadcastTo-position_embedding29/strided_slice_1:output:0#position_embedding29/Shape:output:0*
T0*+
_output_shapes
:????????? ?
addAddV26token_embedding29/embedding_lookup/Identity_1:output:0)position_embedding29/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp$^position_embedding29/ReadVariableOp#^token_embedding29/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding29/ReadVariableOp#position_embedding29/ReadVariableOp2H
"token_embedding29/embedding_lookup"token_embedding29/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?5
D__inference_model_12_layer_call_and_return_conditional_losses_765078
inputs_0
inputs_1U
Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_13_string_lookup_13_equal_y5
1text_vectorization_13_string_lookup_13_selectv2_t	\
Itoken_and_position_embedding_27_token_embedding28_embedding_lookup_764671:	?^
Ltoken_and_position_embedding_27_position_embedding28_readvariableop_resource: [
Itoken_and_position_embedding_28_token_embedding29_embedding_lookup_764695:^
Ltoken_and_position_embedding_28_position_embedding29_readvariableop_resource: m
Wtransformer_encoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource:_
Mtransformer_encoder_23_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_encoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource:]
Ktransformer_encoder_23_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_encoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource:_
Mtransformer_encoder_23_multi_head_attention_value_add_readvariableop_resource:x
btransformer_encoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:f
Xtransformer_encoder_23_multi_head_attention_attention_output_add_readvariableop_resource:^
Ptransformer_encoder_23_layer_normalization_batchnorm_mul_readvariableop_resource:Z
Ltransformer_encoder_23_layer_normalization_batchnorm_readvariableop_resource:P
>transformer_encoder_23_dense_tensordot_readvariableop_resource: J
<transformer_encoder_23_dense_biasadd_readvariableop_resource: R
@transformer_encoder_23_dense_1_tensordot_readvariableop_resource: L
>transformer_encoder_23_dense_1_biasadd_readvariableop_resource:`
Rtransformer_encoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource:\
Ntransformer_encoder_23_layer_normalization_1_batchnorm_readvariableop_resource:m
Wtransformer_decoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource:_
Mtransformer_decoder_23_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_decoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource:]
Ktransformer_decoder_23_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_decoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource:_
Mtransformer_decoder_23_multi_head_attention_value_add_readvariableop_resource:x
btransformer_decoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:f
Xtransformer_decoder_23_multi_head_attention_attention_output_add_readvariableop_resource:^
Ptransformer_decoder_23_layer_normalization_batchnorm_mul_readvariableop_resource:Z
Ltransformer_decoder_23_layer_normalization_batchnorm_readvariableop_resource:P
>transformer_decoder_23_dense_tensordot_readvariableop_resource: J
<transformer_decoder_23_dense_biasadd_readvariableop_resource: R
@transformer_decoder_23_dense_1_tensordot_readvariableop_resource: L
>transformer_decoder_23_dense_1_biasadd_readvariableop_resource:`
Rtransformer_decoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource:\
Ntransformer_decoder_23_layer_normalization_1_batchnorm_readvariableop_resource:9
'dense_23_matmul_readvariableop_resource:@6
(dense_23_biasadd_readvariableop_resource:@9
'dense_24_matmul_readvariableop_resource:@6
(dense_24_biasadd_readvariableop_resource:
identity??dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?Ctoken_and_position_embedding_27/position_embedding28/ReadVariableOp?Btoken_and_position_embedding_27/token_embedding28/embedding_lookup?Ctoken_and_position_embedding_28/position_embedding29/ReadVariableOp?Btoken_and_position_embedding_28/token_embedding29/embedding_lookup?3transformer_decoder_23/dense/BiasAdd/ReadVariableOp?5transformer_decoder_23/dense/Tensordot/ReadVariableOp?5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp?7transformer_decoder_23/dense_1/Tensordot/ReadVariableOp?Ctransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_decoder_23/multi_head_attention/key/add/ReadVariableOp?Ltransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOp?Ntransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOp?Ntransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp?3transformer_encoder_23/dense/BiasAdd/ReadVariableOp?5transformer_encoder_23/dense/Tensordot/ReadVariableOp?5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp?7transformer_encoder_23/dense_1/Tensordot/ReadVariableOp?Ctransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_encoder_23/multi_head_attention/key/add/ReadVariableOp?Ltransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOp?Ntransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOp?Ntransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
text_vectorization_13/SqueezeSqueezeinputs_0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????h
'text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2&text_vectorization_13/Squeeze:output:00text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
5text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/text_vectorization_13/StringSplit/strided_sliceStridedSlice9text_vectorization_13/StringSplit/StringSplitV2:indices:0>text_vectorization_13/StringSplit/strided_slice/stack:output:0@text_vectorization_13/StringSplit/strided_slice/stack_1:output:0@text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
7text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1text_vectorization_13/StringSplit/strided_slice_1StridedSlice7text_vectorization_13/StringSplit/StringSplitV2:shape:0@text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Xtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ftext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0stext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
etext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountmtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ctext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle8text_vectorization_13/StringSplit/StringSplitV2:values:0Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_13/string_lookup_13/EqualEqual8text_vectorization_13/StringSplit/StringSplitV2:values:0.text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/SelectV2SelectV20text_vectorization_13/string_lookup_13/Equal:z:01text_vectorization_13_string_lookup_13_selectv2_tMtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/IdentityIdentity8text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????t
2text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
*text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
9text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_13/RaggedToTensor/Const:output:08text_vectorization_13/string_lookup_13/Identity:output:0;text_vectorization_13/RaggedToTensor/default_value:output:0:text_vectorization_13/StringSplit/strided_slice_1:output:08text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Btoken_and_position_embedding_27/token_embedding28/embedding_lookupResourceGatherItoken_and_position_embedding_27_token_embedding28_embedding_lookup_764671Btext_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*\
_classR
PNloc:@token_and_position_embedding_27/token_embedding28/embedding_lookup/764671*+
_output_shapes
:????????? *
dtype0?
Ktoken_and_position_embedding_27/token_embedding28/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_27/token_embedding28/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_27/token_embedding28/embedding_lookup/764671*+
_output_shapes
:????????? ?
Mtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
:token_and_position_embedding_27/position_embedding28/ShapeShapeVtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_27/position_embedding28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_27/position_embedding28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_27/position_embedding28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_27/position_embedding28/strided_sliceStridedSliceCtoken_and_position_embedding_27/position_embedding28/Shape:output:0Qtoken_and_position_embedding_27/position_embedding28/strided_slice/stack:output:0Stoken_and_position_embedding_27/position_embedding28/strided_slice/stack_1:output:0Stoken_and_position_embedding_27/position_embedding28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_27/position_embedding28/ReadVariableOpReadVariableOpLtoken_and_position_embedding_27_position_embedding28_readvariableop_resource*
_output_shapes

: *
dtype0|
:token_and_position_embedding_27/position_embedding28/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_27/position_embedding28/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_27/position_embedding28/strided_slice_1/stackPackCtoken_and_position_embedding_27/position_embedding28/Const:output:0Utoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1PackKtoken_and_position_embedding_27/position_embedding28/strided_slice:output:0Wtoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2PackEtoken_and_position_embedding_27/position_embedding28/Const_1:output:0Wtoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_27/position_embedding28/strided_slice_1StridedSliceKtoken_and_position_embedding_27/position_embedding28/ReadVariableOp:value:0Stoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack:output:0Utoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_27/position_embedding28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
@token_and_position_embedding_27/position_embedding28/BroadcastToBroadcastToMtoken_and_position_embedding_27/position_embedding28/strided_slice_1:output:0Ctoken_and_position_embedding_27/position_embedding28/Shape:output:0*
T0*+
_output_shapes
:????????? ?
#token_and_position_embedding_27/addAddV2Vtoken_and_position_embedding_27/token_embedding28/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_27/position_embedding28/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? ?
6token_and_position_embedding_28/token_embedding29/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
Btoken_and_position_embedding_28/token_embedding29/embedding_lookupResourceGatherItoken_and_position_embedding_28_token_embedding29_embedding_lookup_764695:token_and_position_embedding_28/token_embedding29/Cast:y:0*
Tindices0*\
_classR
PNloc:@token_and_position_embedding_28/token_embedding29/embedding_lookup/764695*+
_output_shapes
:????????? *
dtype0?
Ktoken_and_position_embedding_28/token_embedding29/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_28/token_embedding29/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_28/token_embedding29/embedding_lookup/764695*+
_output_shapes
:????????? ?
Mtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
:token_and_position_embedding_28/position_embedding29/ShapeShapeVtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_28/position_embedding29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_28/position_embedding29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_28/position_embedding29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_28/position_embedding29/strided_sliceStridedSliceCtoken_and_position_embedding_28/position_embedding29/Shape:output:0Qtoken_and_position_embedding_28/position_embedding29/strided_slice/stack:output:0Stoken_and_position_embedding_28/position_embedding29/strided_slice/stack_1:output:0Stoken_and_position_embedding_28/position_embedding29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_28/position_embedding29/ReadVariableOpReadVariableOpLtoken_and_position_embedding_28_position_embedding29_readvariableop_resource*
_output_shapes

: *
dtype0|
:token_and_position_embedding_28/position_embedding29/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_28/position_embedding29/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_28/position_embedding29/strided_slice_1/stackPackCtoken_and_position_embedding_28/position_embedding29/Const:output:0Utoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1PackKtoken_and_position_embedding_28/position_embedding29/strided_slice:output:0Wtoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2PackEtoken_and_position_embedding_28/position_embedding29/Const_1:output:0Wtoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_28/position_embedding29/strided_slice_1StridedSliceKtoken_and_position_embedding_28/position_embedding29/ReadVariableOp:value:0Stoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack:output:0Utoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_28/position_embedding29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
@token_and_position_embedding_28/position_embedding29/BroadcastToBroadcastToMtoken_and_position_embedding_28/position_embedding29/strided_slice_1:output:0Ctoken_and_position_embedding_28/position_embedding29/Shape:output:0*
T0*+
_output_shapes
:????????? ?
#token_and_position_embedding_28/addAddV2Vtoken_and_position_embedding_28/token_embedding29/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_28/position_embedding29/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? ?

add_12/addAddV2'token_and_position_embedding_27/add:z:0'token_and_position_embedding_28/add:z:0*
T0*+
_output_shapes
:????????? ?
Ntransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_encoder_23/multi_head_attention/query/einsum/EinsumEinsumadd_12/add:z:0Vtransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_encoder_23_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_23/multi_head_attention/query/addAddV2Htransformer_encoder_23/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_encoder_23/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_encoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
=transformer_encoder_23/multi_head_attention/key/einsum/EinsumEinsumadd_12/add:z:0Ttransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_encoder_23/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_encoder_23_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_encoder_23/multi_head_attention/key/addAddV2Ftransformer_encoder_23/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_encoder_23/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_encoder_23/multi_head_attention/value/einsum/EinsumEinsumadd_12/add:z:0Vtransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_encoder_23_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_23/multi_head_attention/value/addAddV2Htransformer_encoder_23/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_encoder_23/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_encoder_23/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
/transformer_encoder_23/multi_head_attention/MulMul9transformer_encoder_23/multi_head_attention/query/add:z:0:transformer_encoder_23/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_encoder_23/multi_head_attention/einsum/EinsumEinsum7transformer_encoder_23/multi_head_attention/key/add:z:03transformer_encoder_23/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
;transformer_encoder_23/multi_head_attention/softmax/SoftmaxSoftmaxBtransformer_encoder_23/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_encoder_23/multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
Atransformer_encoder_23/multi_head_attention/dropout_2/dropout/MulMulEtransformer_encoder_23/multi_head_attention/softmax/Softmax:softmax:0Ltransformer_encoder_23/multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_encoder_23/multi_head_attention/dropout_2/dropout/ShapeShapeEtransformer_encoder_23/multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Ztransformer_encoder_23/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniformLtransformer_encoder_23/multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0?
Ltransformer_encoder_23/multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Jtransformer_encoder_23/multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualctransformer_encoder_23/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0Utransformer_encoder_23/multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_encoder_23/multi_head_attention/dropout_2/dropout/CastCastNtransformer_encoder_23/multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
Ctransformer_encoder_23/multi_head_attention/dropout_2/dropout/Mul_1MulEtransformer_encoder_23/multi_head_attention/dropout_2/dropout/Mul:z:0Ftransformer_encoder_23/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
;transformer_encoder_23/multi_head_attention/einsum_1/EinsumEinsumGtransformer_encoder_23/multi_head_attention/dropout_2/dropout/Mul_1:z:09transformer_encoder_23/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_encoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Jtransformer_encoder_23/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_encoder_23/multi_head_attention/einsum_1/Einsum:output:0atransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Otransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_encoder_23_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
@transformer_encoder_23/multi_head_attention/attention_output/addAddV2Stransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? q
,transformer_encoder_23/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*transformer_encoder_23/dropout/dropout/MulMulDtransformer_encoder_23/multi_head_attention/attention_output/add:z:05transformer_encoder_23/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
,transformer_encoder_23/dropout/dropout/ShapeShapeDtransformer_encoder_23/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
Ctransformer_encoder_23/dropout/dropout/random_uniform/RandomUniformRandomUniform5transformer_encoder_23/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0z
5transformer_encoder_23/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3transformer_encoder_23/dropout/dropout/GreaterEqualGreaterEqualLtransformer_encoder_23/dropout/dropout/random_uniform/RandomUniform:output:0>transformer_encoder_23/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
+transformer_encoder_23/dropout/dropout/CastCast7transformer_encoder_23/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
,transformer_encoder_23/dropout/dropout/Mul_1Mul.transformer_encoder_23/dropout/dropout/Mul:z:0/transformer_encoder_23/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_23/addAddV2add_12/add:z:00transformer_encoder_23/dropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_encoder_23/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_encoder_23/layer_normalization/moments/meanMeantransformer_encoder_23/add:z:0Rtransformer_encoder_23/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_encoder_23/layer_normalization/moments/StopGradientStopGradient@transformer_encoder_23/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_encoder_23/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_23/add:z:0Htransformer_encoder_23/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Mtransformer_encoder_23/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_encoder_23/layer_normalization/moments/varianceMeanHtransformer_encoder_23/layer_normalization/moments/SquaredDifference:z:0Vtransformer_encoder_23/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_encoder_23/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_encoder_23/layer_normalization/batchnorm/addAddV2Dtransformer_encoder_23/layer_normalization/moments/variance:output:0Ctransformer_encoder_23/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/RsqrtRsqrt<transformer_encoder_23/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_encoder_23_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_encoder_23/layer_normalization/batchnorm/mulMul>transformer_encoder_23/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/mul_1Multransformer_encoder_23/add:z:0<transformer_encoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/mul_2Mul@transformer_encoder_23/layer_normalization/moments/mean:output:0<transformer_encoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_encoder_23_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_encoder_23/layer_normalization/batchnorm/subSubKtransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_encoder_23/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_23/layer_normalization/batchnorm/add_1AddV2>transformer_encoder_23/layer_normalization/batchnorm/mul_1:z:0<transformer_encoder_23/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
5transformer_encoder_23/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_encoder_23_dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0u
+transformer_encoder_23/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_encoder_23/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_encoder_23/dense/Tensordot/ShapeShape>transformer_encoder_23/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_encoder_23/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_23/dense/Tensordot/GatherV2GatherV25transformer_encoder_23/dense/Tensordot/Shape:output:04transformer_encoder_23/dense/Tensordot/free:output:0=transformer_encoder_23/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_encoder_23/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_23/dense/Tensordot/GatherV2_1GatherV25transformer_encoder_23/dense/Tensordot/Shape:output:04transformer_encoder_23/dense/Tensordot/axes:output:0?transformer_encoder_23/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_encoder_23/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_encoder_23/dense/Tensordot/ProdProd8transformer_encoder_23/dense/Tensordot/GatherV2:output:05transformer_encoder_23/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_encoder_23/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_23/dense/Tensordot/Prod_1Prod:transformer_encoder_23/dense/Tensordot/GatherV2_1:output:07transformer_encoder_23/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_encoder_23/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_encoder_23/dense/Tensordot/concatConcatV24transformer_encoder_23/dense/Tensordot/free:output:04transformer_encoder_23/dense/Tensordot/axes:output:0;transformer_encoder_23/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_encoder_23/dense/Tensordot/stackPack4transformer_encoder_23/dense/Tensordot/Prod:output:06transformer_encoder_23/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_encoder_23/dense/Tensordot/transpose	Transpose>transformer_encoder_23/layer_normalization/batchnorm/add_1:z:06transformer_encoder_23/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
.transformer_encoder_23/dense/Tensordot/ReshapeReshape4transformer_encoder_23/dense/Tensordot/transpose:y:05transformer_encoder_23/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_encoder_23/dense/Tensordot/MatMulMatMul7transformer_encoder_23/dense/Tensordot/Reshape:output:0=transformer_encoder_23/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? x
.transformer_encoder_23/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: v
4transformer_encoder_23/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_23/dense/Tensordot/concat_1ConcatV28transformer_encoder_23/dense/Tensordot/GatherV2:output:07transformer_encoder_23/dense/Tensordot/Const_2:output:0=transformer_encoder_23/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_encoder_23/dense/TensordotReshape7transformer_encoder_23/dense/Tensordot/MatMul:product:08transformer_encoder_23/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
3transformer_encoder_23/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_23_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
$transformer_encoder_23/dense/BiasAddBiasAdd/transformer_encoder_23/dense/Tensordot:output:0;transformer_encoder_23/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
!transformer_encoder_23/dense/ReluRelu-transformer_encoder_23/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
7transformer_encoder_23/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_encoder_23_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0w
-transformer_encoder_23/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_encoder_23/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_encoder_23/dense_1/Tensordot/ShapeShape/transformer_encoder_23/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_encoder_23/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_23/dense_1/Tensordot/GatherV2GatherV27transformer_encoder_23/dense_1/Tensordot/Shape:output:06transformer_encoder_23/dense_1/Tensordot/free:output:0?transformer_encoder_23/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_encoder_23/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_encoder_23/dense_1/Tensordot/GatherV2_1GatherV27transformer_encoder_23/dense_1/Tensordot/Shape:output:06transformer_encoder_23/dense_1/Tensordot/axes:output:0Atransformer_encoder_23/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_encoder_23/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_23/dense_1/Tensordot/ProdProd:transformer_encoder_23/dense_1/Tensordot/GatherV2:output:07transformer_encoder_23/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_encoder_23/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_encoder_23/dense_1/Tensordot/Prod_1Prod<transformer_encoder_23/dense_1/Tensordot/GatherV2_1:output:09transformer_encoder_23/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_encoder_23/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_23/dense_1/Tensordot/concatConcatV26transformer_encoder_23/dense_1/Tensordot/free:output:06transformer_encoder_23/dense_1/Tensordot/axes:output:0=transformer_encoder_23/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_encoder_23/dense_1/Tensordot/stackPack6transformer_encoder_23/dense_1/Tensordot/Prod:output:08transformer_encoder_23/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_encoder_23/dense_1/Tensordot/transpose	Transpose/transformer_encoder_23/dense/Relu:activations:08transformer_encoder_23/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
0transformer_encoder_23/dense_1/Tensordot/ReshapeReshape6transformer_encoder_23/dense_1/Tensordot/transpose:y:07transformer_encoder_23/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_encoder_23/dense_1/Tensordot/MatMulMatMul9transformer_encoder_23/dense_1/Tensordot/Reshape:output:0?transformer_encoder_23/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
0transformer_encoder_23/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:x
6transformer_encoder_23/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_23/dense_1/Tensordot/concat_1ConcatV2:transformer_encoder_23/dense_1/Tensordot/GatherV2:output:09transformer_encoder_23/dense_1/Tensordot/Const_2:output:0?transformer_encoder_23/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_encoder_23/dense_1/TensordotReshape9transformer_encoder_23/dense_1/Tensordot/MatMul:product:0:transformer_encoder_23/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_encoder_23_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&transformer_encoder_23/dense_1/BiasAddBiasAdd1transformer_encoder_23/dense_1/Tensordot:output:0=transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? s
.transformer_encoder_23/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
,transformer_encoder_23/dropout_1/dropout/MulMul/transformer_encoder_23/dense_1/BiasAdd:output:07transformer_encoder_23/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
.transformer_encoder_23/dropout_1/dropout/ShapeShape/transformer_encoder_23/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
Etransformer_encoder_23/dropout_1/dropout/random_uniform/RandomUniformRandomUniform7transformer_encoder_23/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0|
7transformer_encoder_23/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
5transformer_encoder_23/dropout_1/dropout/GreaterEqualGreaterEqualNtransformer_encoder_23/dropout_1/dropout/random_uniform/RandomUniform:output:0@transformer_encoder_23/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_encoder_23/dropout_1/dropout/CastCast9transformer_encoder_23/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
.transformer_encoder_23/dropout_1/dropout/Mul_1Mul0transformer_encoder_23/dropout_1/dropout/Mul:z:01transformer_encoder_23/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_23/add_1AddV2>transformer_encoder_23/layer_normalization/batchnorm/add_1:z:02transformer_encoder_23/dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ?
Ktransformer_encoder_23/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_encoder_23/layer_normalization_1/moments/meanMean transformer_encoder_23/add_1:z:0Ttransformer_encoder_23/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_encoder_23/layer_normalization_1/moments/StopGradientStopGradientBtransformer_encoder_23/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_encoder_23/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_encoder_23/add_1:z:0Jtransformer_encoder_23/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Otransformer_encoder_23/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_encoder_23/layer_normalization_1/moments/varianceMeanJtransformer_encoder_23/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_encoder_23/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_encoder_23/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_encoder_23/layer_normalization_1/batchnorm/addAddV2Ftransformer_encoder_23/layer_normalization_1/moments/variance:output:0Etransformer_encoder_23/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_encoder_23/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_encoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_encoder_23/layer_normalization_1/batchnorm/mulMul@transformer_encoder_23/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/mul_1Mul transformer_encoder_23/add_1:z:0>transformer_encoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/mul_2MulBtransformer_encoder_23/layer_normalization_1/moments/mean:output:0>transformer_encoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Etransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_encoder_23_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_encoder_23/layer_normalization_1/batchnorm/subSubMtransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_encoder_23/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_23/layer_normalization_1/batchnorm/add_1AddV2@transformer_encoder_23/layer_normalization_1/batchnorm/mul_1:z:0>transformer_encoder_23/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_23/ShapeShape@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:t
*transformer_decoder_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,transformer_decoder_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,transformer_decoder_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$transformer_decoder_23/strided_sliceStridedSlice%transformer_decoder_23/Shape:output:03transformer_decoder_23/strided_slice/stack:output:05transformer_decoder_23/strided_slice/stack_1:output:05transformer_decoder_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_23/strided_slice_1StridedSlice%transformer_decoder_23/Shape:output:05transformer_decoder_23/strided_slice_1/stack:output:07transformer_decoder_23/strided_slice_1/stack_1:output:07transformer_decoder_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"transformer_decoder_23/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"transformer_decoder_23/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_23/rangeRange+transformer_decoder_23/range/start:output:0/transformer_decoder_23/strided_slice_1:output:0+transformer_decoder_23/range/delta:output:0*
_output_shapes
: }
,transformer_decoder_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&transformer_decoder_23/strided_slice_2StridedSlice%transformer_decoder_23/range:output:05transformer_decoder_23/strided_slice_2/stack:output:07transformer_decoder_23/strided_slice_2/stack_1:output:07transformer_decoder_23/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskf
$transformer_decoder_23/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : f
$transformer_decoder_23/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_23/range_1Range-transformer_decoder_23/range_1/start:output:0/transformer_decoder_23/strided_slice_1:output:0-transformer_decoder_23/range_1/delta:output:0*
_output_shapes
: ?
#transformer_decoder_23/GreaterEqualGreaterEqual/transformer_decoder_23/strided_slice_2:output:0'transformer_decoder_23/range_1:output:0*
T0*
_output_shapes

:  ?
transformer_decoder_23/CastCast'transformer_decoder_23/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  v
,transformer_decoder_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_23/strided_slice_3StridedSlice%transformer_decoder_23/Shape:output:05transformer_decoder_23/strided_slice_3/stack:output:07transformer_decoder_23/strided_slice_3/stack_1:output:07transformer_decoder_23/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_23/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_23/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_23/strided_slice_4StridedSlice%transformer_decoder_23/Shape:output:05transformer_decoder_23/strided_slice_4/stack:output:07transformer_decoder_23/strided_slice_4/stack_1:output:07transformer_decoder_23/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&transformer_decoder_23/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
$transformer_decoder_23/Reshape/shapePack/transformer_decoder_23/Reshape/shape/0:output:0/transformer_decoder_23/strided_slice_3:output:0/transformer_decoder_23/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_23/ReshapeReshapetransformer_decoder_23/Cast:y:0-transformer_decoder_23/Reshape/shape:output:0*
T0*"
_output_shapes
:  p
%transformer_decoder_23/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!transformer_decoder_23/ExpandDims
ExpandDims-transformer_decoder_23/strided_slice:output:0.transformer_decoder_23/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
transformer_decoder_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"      d
"transformer_decoder_23/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
transformer_decoder_23/concatConcatV2*transformer_decoder_23/ExpandDims:output:0%transformer_decoder_23/Const:output:0+transformer_decoder_23/concat/axis:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_23/TileTile'transformer_decoder_23/Reshape:output:0&transformer_decoder_23/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Ntransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_23_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_decoder_23/multi_head_attention/query/einsum/EinsumEinsum@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_decoder_23_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_23/multi_head_attention/query/addAddV2Htransformer_decoder_23/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_decoder_23/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_decoder_23_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
=transformer_decoder_23/multi_head_attention/key/einsum/EinsumEinsum@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0Ttransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_decoder_23/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_decoder_23_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_decoder_23/multi_head_attention/key/addAddV2Ftransformer_decoder_23/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_decoder_23/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_23_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
?transformer_decoder_23/multi_head_attention/value/einsum/EinsumEinsum@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_decoder_23_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_23/multi_head_attention/value/addAddV2Htransformer_decoder_23/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_decoder_23/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_decoder_23/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
/transformer_decoder_23/multi_head_attention/MulMul9transformer_decoder_23/multi_head_attention/query/add:z:0:transformer_decoder_23/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_decoder_23/multi_head_attention/einsum/EinsumEinsum7transformer_decoder_23/multi_head_attention/key/add:z:03transformer_decoder_23/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
:transformer_decoder_23/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6transformer_decoder_23/multi_head_attention/ExpandDims
ExpandDims$transformer_decoder_23/Tile:output:0Ctransformer_decoder_23/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
8transformer_decoder_23/multi_head_attention/softmax/CastCast?transformer_decoder_23/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ~
9transformer_decoder_23/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7transformer_decoder_23/multi_head_attention/softmax/subSubBtransformer_decoder_23/multi_head_attention/softmax/sub/x:output:0<transformer_decoder_23/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ~
9transformer_decoder_23/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
7transformer_decoder_23/multi_head_attention/softmax/mulMul;transformer_decoder_23/multi_head_attention/softmax/sub:z:0Btransformer_decoder_23/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
7transformer_decoder_23/multi_head_attention/softmax/addAddV2Btransformer_decoder_23/multi_head_attention/einsum/Einsum:output:0;transformer_decoder_23/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_23/multi_head_attention/softmax/SoftmaxSoftmax;transformer_decoder_23/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_decoder_23/multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
Atransformer_decoder_23/multi_head_attention/dropout_2/dropout/MulMulEtransformer_decoder_23/multi_head_attention/softmax/Softmax:softmax:0Ltransformer_decoder_23/multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_decoder_23/multi_head_attention/dropout_2/dropout/ShapeShapeEtransformer_decoder_23/multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Ztransformer_decoder_23/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniformLtransformer_decoder_23/multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0?
Ltransformer_decoder_23/multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Jtransformer_decoder_23/multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualctransformer_decoder_23/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0Utransformer_decoder_23/multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_decoder_23/multi_head_attention/dropout_2/dropout/CastCastNtransformer_decoder_23/multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
Ctransformer_decoder_23/multi_head_attention/dropout_2/dropout/Mul_1MulEtransformer_decoder_23/multi_head_attention/dropout_2/dropout/Mul:z:0Ftransformer_decoder_23/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_23/multi_head_attention/einsum_1/EinsumEinsumGtransformer_decoder_23/multi_head_attention/dropout_2/dropout/Mul_1:z:09transformer_decoder_23/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_decoder_23_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Jtransformer_decoder_23/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_decoder_23/multi_head_attention/einsum_1/Einsum:output:0atransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Otransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_decoder_23_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
@transformer_decoder_23/multi_head_attention/attention_output/addAddV2Stransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? q
,transformer_decoder_23/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*transformer_decoder_23/dropout/dropout/MulMulDtransformer_decoder_23/multi_head_attention/attention_output/add:z:05transformer_decoder_23/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
,transformer_decoder_23/dropout/dropout/ShapeShapeDtransformer_decoder_23/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
Ctransformer_decoder_23/dropout/dropout/random_uniform/RandomUniformRandomUniform5transformer_decoder_23/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0z
5transformer_decoder_23/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3transformer_decoder_23/dropout/dropout/GreaterEqualGreaterEqualLtransformer_decoder_23/dropout/dropout/random_uniform/RandomUniform:output:0>transformer_decoder_23/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
+transformer_decoder_23/dropout/dropout/CastCast7transformer_decoder_23/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
,transformer_decoder_23/dropout/dropout/Mul_1Mul.transformer_decoder_23/dropout/dropout/Mul:z:0/transformer_decoder_23/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_23/addAddV20transformer_decoder_23/dropout/dropout/Mul_1:z:0@transformer_encoder_23/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_decoder_23/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_decoder_23/layer_normalization/moments/meanMeantransformer_decoder_23/add:z:0Rtransformer_decoder_23/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_decoder_23/layer_normalization/moments/StopGradientStopGradient@transformer_decoder_23/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_decoder_23/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_23/add:z:0Htransformer_decoder_23/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Mtransformer_decoder_23/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_decoder_23/layer_normalization/moments/varianceMeanHtransformer_decoder_23/layer_normalization/moments/SquaredDifference:z:0Vtransformer_decoder_23/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_decoder_23/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_decoder_23/layer_normalization/batchnorm/addAddV2Dtransformer_decoder_23/layer_normalization/moments/variance:output:0Ctransformer_decoder_23/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/RsqrtRsqrt<transformer_decoder_23/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_decoder_23_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_decoder_23/layer_normalization/batchnorm/mulMul>transformer_decoder_23/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/mul_1Multransformer_decoder_23/add:z:0<transformer_decoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/mul_2Mul@transformer_decoder_23/layer_normalization/moments/mean:output:0<transformer_decoder_23/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_decoder_23_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
8transformer_decoder_23/layer_normalization/batchnorm/subSubKtransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_decoder_23/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_23/layer_normalization/batchnorm/add_1AddV2>transformer_decoder_23/layer_normalization/batchnorm/mul_1:z:0<transformer_decoder_23/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
5transformer_decoder_23/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_decoder_23_dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0u
+transformer_decoder_23/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_decoder_23/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_decoder_23/dense/Tensordot/ShapeShape>transformer_decoder_23/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_decoder_23/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_23/dense/Tensordot/GatherV2GatherV25transformer_decoder_23/dense/Tensordot/Shape:output:04transformer_decoder_23/dense/Tensordot/free:output:0=transformer_decoder_23/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_decoder_23/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_23/dense/Tensordot/GatherV2_1GatherV25transformer_decoder_23/dense/Tensordot/Shape:output:04transformer_decoder_23/dense/Tensordot/axes:output:0?transformer_decoder_23/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_decoder_23/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_decoder_23/dense/Tensordot/ProdProd8transformer_decoder_23/dense/Tensordot/GatherV2:output:05transformer_decoder_23/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_decoder_23/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_23/dense/Tensordot/Prod_1Prod:transformer_decoder_23/dense/Tensordot/GatherV2_1:output:07transformer_decoder_23/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_decoder_23/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_decoder_23/dense/Tensordot/concatConcatV24transformer_decoder_23/dense/Tensordot/free:output:04transformer_decoder_23/dense/Tensordot/axes:output:0;transformer_decoder_23/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_decoder_23/dense/Tensordot/stackPack4transformer_decoder_23/dense/Tensordot/Prod:output:06transformer_decoder_23/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_decoder_23/dense/Tensordot/transpose	Transpose>transformer_decoder_23/layer_normalization/batchnorm/add_1:z:06transformer_decoder_23/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
.transformer_decoder_23/dense/Tensordot/ReshapeReshape4transformer_decoder_23/dense/Tensordot/transpose:y:05transformer_decoder_23/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_decoder_23/dense/Tensordot/MatMulMatMul7transformer_decoder_23/dense/Tensordot/Reshape:output:0=transformer_decoder_23/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? x
.transformer_decoder_23/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: v
4transformer_decoder_23/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_23/dense/Tensordot/concat_1ConcatV28transformer_decoder_23/dense/Tensordot/GatherV2:output:07transformer_decoder_23/dense/Tensordot/Const_2:output:0=transformer_decoder_23/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_decoder_23/dense/TensordotReshape7transformer_decoder_23/dense/Tensordot/MatMul:product:08transformer_decoder_23/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
3transformer_decoder_23/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_decoder_23_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
$transformer_decoder_23/dense/BiasAddBiasAdd/transformer_decoder_23/dense/Tensordot:output:0;transformer_decoder_23/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
!transformer_decoder_23/dense/ReluRelu-transformer_decoder_23/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
7transformer_decoder_23/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_decoder_23_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0w
-transformer_decoder_23/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_decoder_23/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_decoder_23/dense_1/Tensordot/ShapeShape/transformer_decoder_23/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_decoder_23/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_23/dense_1/Tensordot/GatherV2GatherV27transformer_decoder_23/dense_1/Tensordot/Shape:output:06transformer_decoder_23/dense_1/Tensordot/free:output:0?transformer_decoder_23/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_decoder_23/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_decoder_23/dense_1/Tensordot/GatherV2_1GatherV27transformer_decoder_23/dense_1/Tensordot/Shape:output:06transformer_decoder_23/dense_1/Tensordot/axes:output:0Atransformer_decoder_23/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_decoder_23/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_23/dense_1/Tensordot/ProdProd:transformer_decoder_23/dense_1/Tensordot/GatherV2:output:07transformer_decoder_23/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_decoder_23/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_decoder_23/dense_1/Tensordot/Prod_1Prod<transformer_decoder_23/dense_1/Tensordot/GatherV2_1:output:09transformer_decoder_23/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_decoder_23/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_23/dense_1/Tensordot/concatConcatV26transformer_decoder_23/dense_1/Tensordot/free:output:06transformer_decoder_23/dense_1/Tensordot/axes:output:0=transformer_decoder_23/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_decoder_23/dense_1/Tensordot/stackPack6transformer_decoder_23/dense_1/Tensordot/Prod:output:08transformer_decoder_23/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_decoder_23/dense_1/Tensordot/transpose	Transpose/transformer_decoder_23/dense/Relu:activations:08transformer_decoder_23/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
0transformer_decoder_23/dense_1/Tensordot/ReshapeReshape6transformer_decoder_23/dense_1/Tensordot/transpose:y:07transformer_decoder_23/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_decoder_23/dense_1/Tensordot/MatMulMatMul9transformer_decoder_23/dense_1/Tensordot/Reshape:output:0?transformer_decoder_23/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
0transformer_decoder_23/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:x
6transformer_decoder_23/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_23/dense_1/Tensordot/concat_1ConcatV2:transformer_decoder_23/dense_1/Tensordot/GatherV2:output:09transformer_decoder_23/dense_1/Tensordot/Const_2:output:0?transformer_decoder_23/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_decoder_23/dense_1/TensordotReshape9transformer_decoder_23/dense_1/Tensordot/MatMul:product:0:transformer_decoder_23/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_decoder_23_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
&transformer_decoder_23/dense_1/BiasAddBiasAdd1transformer_decoder_23/dense_1/Tensordot:output:0=transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? s
.transformer_decoder_23/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
,transformer_decoder_23/dropout_1/dropout/MulMul/transformer_decoder_23/dense_1/BiasAdd:output:07transformer_decoder_23/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
.transformer_decoder_23/dropout_1/dropout/ShapeShape/transformer_decoder_23/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
Etransformer_decoder_23/dropout_1/dropout/random_uniform/RandomUniformRandomUniform7transformer_decoder_23/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0|
7transformer_decoder_23/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
5transformer_decoder_23/dropout_1/dropout/GreaterEqualGreaterEqualNtransformer_decoder_23/dropout_1/dropout/random_uniform/RandomUniform:output:0@transformer_decoder_23/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_decoder_23/dropout_1/dropout/CastCast9transformer_decoder_23/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
.transformer_decoder_23/dropout_1/dropout/Mul_1Mul0transformer_decoder_23/dropout_1/dropout/Mul:z:01transformer_decoder_23/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_23/add_1AddV2>transformer_decoder_23/layer_normalization/batchnorm/add_1:z:02transformer_decoder_23/dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ?
Ktransformer_decoder_23/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_decoder_23/layer_normalization_1/moments/meanMean transformer_decoder_23/add_1:z:0Ttransformer_decoder_23/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_decoder_23/layer_normalization_1/moments/StopGradientStopGradientBtransformer_decoder_23/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_decoder_23/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_decoder_23/add_1:z:0Jtransformer_decoder_23/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Otransformer_decoder_23/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_decoder_23/layer_normalization_1/moments/varianceMeanJtransformer_decoder_23/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_decoder_23/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_decoder_23/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_decoder_23/layer_normalization_1/batchnorm/addAddV2Ftransformer_decoder_23/layer_normalization_1/moments/variance:output:0Etransformer_decoder_23/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_decoder_23/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_decoder_23_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_decoder_23/layer_normalization_1/batchnorm/mulMul@transformer_decoder_23/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/mul_1Mul transformer_decoder_23/add_1:z:0>transformer_decoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/mul_2MulBtransformer_decoder_23/layer_normalization_1/moments/mean:output:0>transformer_decoder_23/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Etransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_decoder_23_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
:transformer_decoder_23/layer_normalization_1/batchnorm/subSubMtransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_decoder_23/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_23/layer_normalization_1/batchnorm/add_1AddV2@transformer_decoder_23/layer_normalization_1/batchnorm/mul_1:z:0>transformer_decoder_23/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? t
2global_average_pooling1d_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
 global_average_pooling1d_12/MeanMean@transformer_decoder_23/layer_normalization_1/batchnorm/add_1:z:0;global_average_pooling1d_12/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_23/MatMulMatMul)global_average_pooling1d_12/Mean:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@]
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_10/dropout/MulMuldense_23/Relu:activations:0!dropout_10/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@c
dropout_10/dropout/ShapeShapedense_23/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0f
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@?
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@?
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_24/MatMulMatMuldropout_10/dropout/Mul_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_24/SigmoidSigmoiddense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????c
IdentityIdentitydense_24/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOpE^text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2D^token_and_position_embedding_27/position_embedding28/ReadVariableOpC^token_and_position_embedding_27/token_embedding28/embedding_lookupD^token_and_position_embedding_28/position_embedding29/ReadVariableOpC^token_and_position_embedding_28/token_embedding29/embedding_lookup4^transformer_decoder_23/dense/BiasAdd/ReadVariableOp6^transformer_decoder_23/dense/Tensordot/ReadVariableOp6^transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp8^transformer_decoder_23/dense_1/Tensordot/ReadVariableOpD^transformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpH^transformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_decoder_23/multi_head_attention/key/add/ReadVariableOpM^transformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_decoder_23/multi_head_attention/query/add/ReadVariableOpO^transformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_decoder_23/multi_head_attention/value/add/ReadVariableOpO^transformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp4^transformer_encoder_23/dense/BiasAdd/ReadVariableOp6^transformer_encoder_23/dense/Tensordot/ReadVariableOp6^transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp8^transformer_encoder_23/dense_1/Tensordot/ReadVariableOpD^transformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpH^transformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_encoder_23/multi_head_attention/key/add/ReadVariableOpM^transformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_encoder_23/multi_head_attention/query/add/ReadVariableOpO^transformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_encoder_23/multi_head_attention/value/add/ReadVariableOpO^transformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2?
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22?
Ctoken_and_position_embedding_27/position_embedding28/ReadVariableOpCtoken_and_position_embedding_27/position_embedding28/ReadVariableOp2?
Btoken_and_position_embedding_27/token_embedding28/embedding_lookupBtoken_and_position_embedding_27/token_embedding28/embedding_lookup2?
Ctoken_and_position_embedding_28/position_embedding29/ReadVariableOpCtoken_and_position_embedding_28/position_embedding29/ReadVariableOp2?
Btoken_and_position_embedding_28/token_embedding29/embedding_lookupBtoken_and_position_embedding_28/token_embedding29/embedding_lookup2j
3transformer_decoder_23/dense/BiasAdd/ReadVariableOp3transformer_decoder_23/dense/BiasAdd/ReadVariableOp2n
5transformer_decoder_23/dense/Tensordot/ReadVariableOp5transformer_decoder_23/dense/Tensordot/ReadVariableOp2n
5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp5transformer_decoder_23/dense_1/BiasAdd/ReadVariableOp2r
7transformer_decoder_23/dense_1/Tensordot/ReadVariableOp7transformer_decoder_23/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOpCtransformer_decoder_23/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_decoder_23/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_decoder_23/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_decoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_decoder_23/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_decoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_decoder_23/multi_head_attention/key/add/ReadVariableOpBtransformer_decoder_23/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_decoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOpDtransformer_decoder_23/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_decoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOpDtransformer_decoder_23/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_decoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp2j
3transformer_encoder_23/dense/BiasAdd/ReadVariableOp3transformer_encoder_23/dense/BiasAdd/ReadVariableOp2n
5transformer_encoder_23/dense/Tensordot/ReadVariableOp5transformer_encoder_23/dense/Tensordot/ReadVariableOp2n
5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp5transformer_encoder_23/dense_1/BiasAdd/ReadVariableOp2r
7transformer_encoder_23/dense_1/Tensordot/ReadVariableOp7transformer_encoder_23/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOpCtransformer_encoder_23/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_encoder_23/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_encoder_23/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_encoder_23/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_encoder_23/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_encoder_23/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_encoder_23/multi_head_attention/key/add/ReadVariableOpBtransformer_encoder_23/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_encoder_23/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOpDtransformer_encoder_23/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_encoder_23/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOpDtransformer_encoder_23/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_encoder_23/multi_head_attention/value/einsum/Einsum/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_23_layer_call_and_return_conditional_losses_765984

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_7660449
5key_value_init614409_lookuptableimportv2_table_handle1
-key_value_init614409_lookuptableimportv2_keys3
/key_value_init614409_lookuptableimportv2_values	
identity??(key_value_init614409/LookupTableImportV2?
(key_value_init614409/LookupTableImportV2LookupTableImportV25key_value_init614409_lookuptableimportv2_table_handle-key_value_init614409_lookuptableimportv2_keys/key_value_init614409_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^key_value_init614409/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2T
(key_value_init614409/LookupTableImportV2(key_value_init614409/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?	
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_766011

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
/
__inference__initializer_766059
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
??
?
D__inference_model_12_layer_call_and_return_conditional_losses_763729

phrase

token_roleU
Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_13_string_lookup_13_equal_y5
1text_vectorization_13_string_lookup_13_selectv2_t	9
&token_and_position_embedding_27_763639:	?8
&token_and_position_embedding_27_763641: 8
&token_and_position_embedding_28_763644:8
&token_and_position_embedding_28_763646: 3
transformer_encoder_23_763650:/
transformer_encoder_23_763652:3
transformer_encoder_23_763654:/
transformer_encoder_23_763656:3
transformer_encoder_23_763658:/
transformer_encoder_23_763660:3
transformer_encoder_23_763662:+
transformer_encoder_23_763664:+
transformer_encoder_23_763666:+
transformer_encoder_23_763668:/
transformer_encoder_23_763670: +
transformer_encoder_23_763672: /
transformer_encoder_23_763674: +
transformer_encoder_23_763676:+
transformer_encoder_23_763678:+
transformer_encoder_23_763680:3
transformer_decoder_23_763683:/
transformer_decoder_23_763685:3
transformer_decoder_23_763687:/
transformer_decoder_23_763689:3
transformer_decoder_23_763691:/
transformer_decoder_23_763693:3
transformer_decoder_23_763695:+
transformer_decoder_23_763697:+
transformer_decoder_23_763699:+
transformer_decoder_23_763701:/
transformer_decoder_23_763703: +
transformer_decoder_23_763705: /
transformer_decoder_23_763707: +
transformer_decoder_23_763709:+
transformer_decoder_23_763711:+
transformer_decoder_23_763713:!
dense_23_763717:@
dense_23_763719:@!
dense_24_763723:@
dense_24_763725:
identity?? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall?Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?7token_and_position_embedding_27/StatefulPartitionedCall?7token_and_position_embedding_28/StatefulPartitionedCall?.transformer_decoder_23/StatefulPartitionedCall?.transformer_encoder_23/StatefulPartitionedCall~
text_vectorization_13/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????h
'text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2&text_vectorization_13/Squeeze:output:00text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
5text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/text_vectorization_13/StringSplit/strided_sliceStridedSlice9text_vectorization_13/StringSplit/StringSplitV2:indices:0>text_vectorization_13/StringSplit/strided_slice/stack:output:0@text_vectorization_13/StringSplit/strided_slice/stack_1:output:0@text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
7text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1text_vectorization_13/StringSplit/strided_slice_1StridedSlice7text_vectorization_13/StringSplit/StringSplitV2:shape:0@text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Xtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ftext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0stext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
etext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountmtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ctext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle8text_vectorization_13/StringSplit/StringSplitV2:values:0Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_13/string_lookup_13/EqualEqual8text_vectorization_13/StringSplit/StringSplitV2:values:0.text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/SelectV2SelectV20text_vectorization_13/string_lookup_13/Equal:z:01text_vectorization_13_string_lookup_13_selectv2_tMtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/IdentityIdentity8text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????t
2text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
*text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
9text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_13/RaggedToTensor/Const:output:08text_vectorization_13/string_lookup_13/Identity:output:0;text_vectorization_13/RaggedToTensor/default_value:output:0:text_vectorization_13/StringSplit/strided_slice_1:output:08text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_27/StatefulPartitionedCallStatefulPartitionedCallBtext_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_27_763639&token_and_position_embedding_27_763641*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_762051?
7token_and_position_embedding_28/StatefulPartitionedCallStatefulPartitionedCall
token_role&token_and_position_embedding_28_763644&token_and_position_embedding_28_763646*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_762085?
add_12/PartitionedCallPartitionedCall@token_and_position_embedding_27/StatefulPartitionedCall:output:0@token_and_position_embedding_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_762097?
.transformer_encoder_23/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0transformer_encoder_23_763650transformer_encoder_23_763652transformer_encoder_23_763654transformer_encoder_23_763656transformer_encoder_23_763658transformer_encoder_23_763660transformer_encoder_23_763662transformer_encoder_23_763664transformer_encoder_23_763666transformer_encoder_23_763668transformer_encoder_23_763670transformer_encoder_23_763672transformer_encoder_23_763674transformer_encoder_23_763676transformer_encoder_23_763678transformer_encoder_23_763680*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_762226?
.transformer_decoder_23/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_23/StatefulPartitionedCall:output:0transformer_decoder_23_763683transformer_decoder_23_763685transformer_decoder_23_763687transformer_decoder_23_763689transformer_decoder_23_763691transformer_decoder_23_763693transformer_decoder_23_763695transformer_decoder_23_763697transformer_decoder_23_763699transformer_decoder_23_763701transformer_decoder_23_763703transformer_decoder_23_763705transformer_decoder_23_763707transformer_decoder_23_763709transformer_decoder_23_763711transformer_decoder_23_763713*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762433?
+global_average_pooling1d_12/PartitionedCallPartitionedCall7transformer_decoder_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_761964?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0dense_23_763717dense_23_763719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_762479?
dropout_10/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_762490?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_24_763723dense_24_763725*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_762503x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCallE^text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV28^token_and_position_embedding_27/StatefulPartitionedCall8^token_and_position_embedding_28/StatefulPartitionedCall/^transformer_decoder_23/StatefulPartitionedCall/^transformer_encoder_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2?
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_27/StatefulPartitionedCall7token_and_position_embedding_27/StatefulPartitionedCall2r
7token_and_position_embedding_28/StatefulPartitionedCall7token_and_position_embedding_28/StatefulPartitionedCall2`
.transformer_decoder_23/StatefulPartitionedCall.transformer_decoder_23/StatefulPartitionedCall2`
.transformer_encoder_23/StatefulPartitionedCall.transformer_encoder_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namephrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
token_role:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?

)__inference_model_12_layer_call_fn_762601

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:

unknown_37:

unknown_38:

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallphrase
token_roleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_762510o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namephrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
token_role:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762433
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :p
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes
: f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlicerange:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskO
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :v
range_1Rangerange_1/start:output:0strided_slice_1:output:0range_1/delta:output:0*
_output_shapes
: q
GreaterEqualGreaterEqualstrided_slice_2:output:0range_1:output:0*
T0*
_output_shapes

:  V
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0strided_slice_3:output:0strided_slice_4:output:0*
N*
T0*
_output_shapes
:a
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*"
_output_shapes
:  Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concatConcatV2ExpandDims:output:0Const:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:e
TileTileReshape:output:0concat:output:0*
T0*+
_output_shapes
:?????????  ?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acben
#multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
multi_head_attention/ExpandDims
ExpandDimsTile:output:0,multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
!multi_head_attention/softmax/CastCast(multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 multi_head_attention/softmax/subSub+multi_head_attention/softmax/sub/x:output:0%multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
 multi_head_attention/softmax/mulMul$multi_head_attention/softmax/sub:z:0+multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
 multi_head_attention/softmax/addAddV2+multi_head_attention/einsum/Einsum:output:0$multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/softmax/SoftmaxSoftmax$multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
'multi_head_attention/dropout_2/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? o
addAddV2dropout/Identity:output:0decoder_sequence*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:] Y
+
_output_shapes
:????????? 
*
_user_specified_namedecoder_sequence
?

?
'__inference_restore_from_tensors_766753M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: @
<mutablehashtable_table_restore_lookuptableimportv2_restorev2B
>mutablehashtable_table_restore_lookuptableimportv2_restorev2_1	
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable<mutablehashtable_table_restore_lookuptableimportv2_restorev2>mutablehashtable_table_restore_lookuptableimportv2_restorev2_1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes

: ::2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:) %
#
_class
loc:@MutableHashTable:C?
#
_class
loc:@MutableHashTable

_output_shapes
::C?
#
_class
loc:@MutableHashTable

_output_shapes
:
?
n
B__inference_add_12_layer_call_and_return_conditional_losses_765163
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:????????? S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
d
+__inference_dropout_10_layer_call_fn_765994

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_762631o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?o
"__inference__traced_restore_766926
file_prefix2
 assignvariableop_dense_23_kernel:@.
 assignvariableop_1_dense_23_bias:@4
"assignvariableop_2_dense_24_kernel:@.
 assignvariableop_3_dense_24_bias:b
Oassignvariableop_4_token_and_position_embedding_27_token_embedding28_embeddings:	?d
Rassignvariableop_5_token_and_position_embedding_27_position_embedding28_embeddings: a
Oassignvariableop_6_token_and_position_embedding_28_token_embedding29_embeddings:d
Rassignvariableop_7_token_and_position_embedding_28_position_embedding29_embeddings: a
Kassignvariableop_8_transformer_encoder_23_multi_head_attention_query_kernel:[
Iassignvariableop_9_transformer_encoder_23_multi_head_attention_query_bias:`
Jassignvariableop_10_transformer_encoder_23_multi_head_attention_key_kernel:Z
Hassignvariableop_11_transformer_encoder_23_multi_head_attention_key_bias:b
Lassignvariableop_12_transformer_encoder_23_multi_head_attention_value_kernel:\
Jassignvariableop_13_transformer_encoder_23_multi_head_attention_value_bias:m
Wassignvariableop_14_transformer_encoder_23_multi_head_attention_attention_output_kernel:c
Uassignvariableop_15_transformer_encoder_23_multi_head_attention_attention_output_bias:R
Dassignvariableop_16_transformer_encoder_23_layer_normalization_gamma:Q
Cassignvariableop_17_transformer_encoder_23_layer_normalization_beta:T
Fassignvariableop_18_transformer_encoder_23_layer_normalization_1_gamma:S
Eassignvariableop_19_transformer_encoder_23_layer_normalization_1_beta:I
7assignvariableop_20_transformer_encoder_23_dense_kernel: C
5assignvariableop_21_transformer_encoder_23_dense_bias: K
9assignvariableop_22_transformer_encoder_23_dense_1_kernel: E
7assignvariableop_23_transformer_encoder_23_dense_1_bias:b
Lassignvariableop_24_transformer_decoder_23_multi_head_attention_query_kernel:\
Jassignvariableop_25_transformer_decoder_23_multi_head_attention_query_bias:`
Jassignvariableop_26_transformer_decoder_23_multi_head_attention_key_kernel:Z
Hassignvariableop_27_transformer_decoder_23_multi_head_attention_key_bias:b
Lassignvariableop_28_transformer_decoder_23_multi_head_attention_value_kernel:\
Jassignvariableop_29_transformer_decoder_23_multi_head_attention_value_bias:m
Wassignvariableop_30_transformer_decoder_23_multi_head_attention_attention_output_kernel:c
Uassignvariableop_31_transformer_decoder_23_multi_head_attention_attention_output_bias:R
Dassignvariableop_32_transformer_decoder_23_layer_normalization_gamma:Q
Cassignvariableop_33_transformer_decoder_23_layer_normalization_beta:T
Fassignvariableop_34_transformer_decoder_23_layer_normalization_1_gamma:S
Eassignvariableop_35_transformer_decoder_23_layer_normalization_1_beta:I
7assignvariableop_36_transformer_decoder_23_dense_kernel: C
5assignvariableop_37_transformer_decoder_23_dense_bias: K
9assignvariableop_38_transformer_decoder_23_dense_1_kernel: E
7assignvariableop_39_transformer_decoder_23_dense_1_bias:'
assignvariableop_40_adam_iter:	 )
assignvariableop_41_adam_beta_1: )
assignvariableop_42_adam_beta_2: (
assignvariableop_43_adam_decay: 0
&assignvariableop_44_adam_learning_rate: 
mutablehashtable: %
assignvariableop_45_total_1: %
assignvariableop_46_count_1: #
assignvariableop_47_total: #
assignvariableop_48_count: <
*assignvariableop_49_adam_dense_23_kernel_m:@6
(assignvariableop_50_adam_dense_23_bias_m:@<
*assignvariableop_51_adam_dense_24_kernel_m:@6
(assignvariableop_52_adam_dense_24_bias_m:j
Wassignvariableop_53_adam_token_and_position_embedding_27_token_embedding28_embeddings_m:	?l
Zassignvariableop_54_adam_token_and_position_embedding_27_position_embedding28_embeddings_m: i
Wassignvariableop_55_adam_token_and_position_embedding_28_token_embedding29_embeddings_m:l
Zassignvariableop_56_adam_token_and_position_embedding_28_position_embedding29_embeddings_m: i
Sassignvariableop_57_adam_transformer_encoder_23_multi_head_attention_query_kernel_m:c
Qassignvariableop_58_adam_transformer_encoder_23_multi_head_attention_query_bias_m:g
Qassignvariableop_59_adam_transformer_encoder_23_multi_head_attention_key_kernel_m:a
Oassignvariableop_60_adam_transformer_encoder_23_multi_head_attention_key_bias_m:i
Sassignvariableop_61_adam_transformer_encoder_23_multi_head_attention_value_kernel_m:c
Qassignvariableop_62_adam_transformer_encoder_23_multi_head_attention_value_bias_m:t
^assignvariableop_63_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_m:j
\assignvariableop_64_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_m:Y
Kassignvariableop_65_adam_transformer_encoder_23_layer_normalization_gamma_m:X
Jassignvariableop_66_adam_transformer_encoder_23_layer_normalization_beta_m:[
Massignvariableop_67_adam_transformer_encoder_23_layer_normalization_1_gamma_m:Z
Lassignvariableop_68_adam_transformer_encoder_23_layer_normalization_1_beta_m:P
>assignvariableop_69_adam_transformer_encoder_23_dense_kernel_m: J
<assignvariableop_70_adam_transformer_encoder_23_dense_bias_m: R
@assignvariableop_71_adam_transformer_encoder_23_dense_1_kernel_m: L
>assignvariableop_72_adam_transformer_encoder_23_dense_1_bias_m:i
Sassignvariableop_73_adam_transformer_decoder_23_multi_head_attention_query_kernel_m:c
Qassignvariableop_74_adam_transformer_decoder_23_multi_head_attention_query_bias_m:g
Qassignvariableop_75_adam_transformer_decoder_23_multi_head_attention_key_kernel_m:a
Oassignvariableop_76_adam_transformer_decoder_23_multi_head_attention_key_bias_m:i
Sassignvariableop_77_adam_transformer_decoder_23_multi_head_attention_value_kernel_m:c
Qassignvariableop_78_adam_transformer_decoder_23_multi_head_attention_value_bias_m:t
^assignvariableop_79_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_m:j
\assignvariableop_80_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_m:Y
Kassignvariableop_81_adam_transformer_decoder_23_layer_normalization_gamma_m:X
Jassignvariableop_82_adam_transformer_decoder_23_layer_normalization_beta_m:[
Massignvariableop_83_adam_transformer_decoder_23_layer_normalization_1_gamma_m:Z
Lassignvariableop_84_adam_transformer_decoder_23_layer_normalization_1_beta_m:P
>assignvariableop_85_adam_transformer_decoder_23_dense_kernel_m: J
<assignvariableop_86_adam_transformer_decoder_23_dense_bias_m: R
@assignvariableop_87_adam_transformer_decoder_23_dense_1_kernel_m: L
>assignvariableop_88_adam_transformer_decoder_23_dense_1_bias_m:<
*assignvariableop_89_adam_dense_23_kernel_v:@6
(assignvariableop_90_adam_dense_23_bias_v:@<
*assignvariableop_91_adam_dense_24_kernel_v:@6
(assignvariableop_92_adam_dense_24_bias_v:j
Wassignvariableop_93_adam_token_and_position_embedding_27_token_embedding28_embeddings_v:	?l
Zassignvariableop_94_adam_token_and_position_embedding_27_position_embedding28_embeddings_v: i
Wassignvariableop_95_adam_token_and_position_embedding_28_token_embedding29_embeddings_v:l
Zassignvariableop_96_adam_token_and_position_embedding_28_position_embedding29_embeddings_v: i
Sassignvariableop_97_adam_transformer_encoder_23_multi_head_attention_query_kernel_v:c
Qassignvariableop_98_adam_transformer_encoder_23_multi_head_attention_query_bias_v:g
Qassignvariableop_99_adam_transformer_encoder_23_multi_head_attention_key_kernel_v:b
Passignvariableop_100_adam_transformer_encoder_23_multi_head_attention_key_bias_v:j
Tassignvariableop_101_adam_transformer_encoder_23_multi_head_attention_value_kernel_v:d
Rassignvariableop_102_adam_transformer_encoder_23_multi_head_attention_value_bias_v:u
_assignvariableop_103_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_v:k
]assignvariableop_104_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_v:Z
Lassignvariableop_105_adam_transformer_encoder_23_layer_normalization_gamma_v:Y
Kassignvariableop_106_adam_transformer_encoder_23_layer_normalization_beta_v:\
Nassignvariableop_107_adam_transformer_encoder_23_layer_normalization_1_gamma_v:[
Massignvariableop_108_adam_transformer_encoder_23_layer_normalization_1_beta_v:Q
?assignvariableop_109_adam_transformer_encoder_23_dense_kernel_v: K
=assignvariableop_110_adam_transformer_encoder_23_dense_bias_v: S
Aassignvariableop_111_adam_transformer_encoder_23_dense_1_kernel_v: M
?assignvariableop_112_adam_transformer_encoder_23_dense_1_bias_v:j
Tassignvariableop_113_adam_transformer_decoder_23_multi_head_attention_query_kernel_v:d
Rassignvariableop_114_adam_transformer_decoder_23_multi_head_attention_query_bias_v:h
Rassignvariableop_115_adam_transformer_decoder_23_multi_head_attention_key_kernel_v:b
Passignvariableop_116_adam_transformer_decoder_23_multi_head_attention_key_bias_v:j
Tassignvariableop_117_adam_transformer_decoder_23_multi_head_attention_value_kernel_v:d
Rassignvariableop_118_adam_transformer_decoder_23_multi_head_attention_value_bias_v:u
_assignvariableop_119_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_v:k
]assignvariableop_120_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_v:Z
Lassignvariableop_121_adam_transformer_decoder_23_layer_normalization_gamma_v:Y
Kassignvariableop_122_adam_transformer_decoder_23_layer_normalization_beta_v:\
Nassignvariableop_123_adam_transformer_decoder_23_layer_normalization_1_gamma_v:[
Massignvariableop_124_adam_transformer_decoder_23_layer_normalization_1_beta_v:Q
?assignvariableop_125_adam_transformer_decoder_23_dense_kernel_v: K
=assignvariableop_126_adam_transformer_decoder_23_dense_bias_v: S
Aassignvariableop_127_adam_transformer_decoder_23_dense_1_kernel_v: M
?assignvariableop_128_adam_transformer_decoder_23_dense_1_bias_v:
identity_130??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?StatefulPartitionedCall?>
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?>
value?=B?=?B6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_24_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_24_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpOassignvariableop_4_token_and_position_embedding_27_token_embedding28_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpRassignvariableop_5_token_and_position_embedding_27_position_embedding28_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpOassignvariableop_6_token_and_position_embedding_28_token_embedding29_embeddingsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpRassignvariableop_7_token_and_position_embedding_28_position_embedding29_embeddingsIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpKassignvariableop_8_transformer_encoder_23_multi_head_attention_query_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpIassignvariableop_9_transformer_encoder_23_multi_head_attention_query_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpJassignvariableop_10_transformer_encoder_23_multi_head_attention_key_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpHassignvariableop_11_transformer_encoder_23_multi_head_attention_key_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpLassignvariableop_12_transformer_encoder_23_multi_head_attention_value_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpJassignvariableop_13_transformer_encoder_23_multi_head_attention_value_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpWassignvariableop_14_transformer_encoder_23_multi_head_attention_attention_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpUassignvariableop_15_transformer_encoder_23_multi_head_attention_attention_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpDassignvariableop_16_transformer_encoder_23_layer_normalization_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpCassignvariableop_17_transformer_encoder_23_layer_normalization_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpFassignvariableop_18_transformer_encoder_23_layer_normalization_1_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpEassignvariableop_19_transformer_encoder_23_layer_normalization_1_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp7assignvariableop_20_transformer_encoder_23_dense_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_transformer_encoder_23_dense_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp9assignvariableop_22_transformer_encoder_23_dense_1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp7assignvariableop_23_transformer_encoder_23_dense_1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpLassignvariableop_24_transformer_decoder_23_multi_head_attention_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpJassignvariableop_25_transformer_decoder_23_multi_head_attention_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpJassignvariableop_26_transformer_decoder_23_multi_head_attention_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpHassignvariableop_27_transformer_decoder_23_multi_head_attention_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpLassignvariableop_28_transformer_decoder_23_multi_head_attention_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpJassignvariableop_29_transformer_decoder_23_multi_head_attention_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpWassignvariableop_30_transformer_decoder_23_multi_head_attention_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpUassignvariableop_31_transformer_decoder_23_multi_head_attention_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpDassignvariableop_32_transformer_decoder_23_layer_normalization_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpCassignvariableop_33_transformer_decoder_23_layer_normalization_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpFassignvariableop_34_transformer_decoder_23_layer_normalization_1_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpEassignvariableop_35_transformer_decoder_23_layer_normalization_1_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp7assignvariableop_36_transformer_decoder_23_dense_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_transformer_decoder_23_dense_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp9assignvariableop_38_transformer_decoder_23_dense_1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp7assignvariableop_39_transformer_decoder_23_dense_1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_iterIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_decayIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_learning_rateIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
StatefulPartitionedCallStatefulPartitionedCallmutablehashtableRestoreV2:tensors:45RestoreV2:tensors:46"/device:CPU:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *0
f+R)
'__inference_restore_from_tensors_766753_
Identity_45IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_23_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_23_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_24_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_24_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpWassignvariableop_53_adam_token_and_position_embedding_27_token_embedding28_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpZassignvariableop_54_adam_token_and_position_embedding_27_position_embedding28_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpWassignvariableop_55_adam_token_and_position_embedding_28_token_embedding29_embeddings_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpZassignvariableop_56_adam_token_and_position_embedding_28_position_embedding29_embeddings_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpSassignvariableop_57_adam_transformer_encoder_23_multi_head_attention_query_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpQassignvariableop_58_adam_transformer_encoder_23_multi_head_attention_query_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpQassignvariableop_59_adam_transformer_encoder_23_multi_head_attention_key_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpOassignvariableop_60_adam_transformer_encoder_23_multi_head_attention_key_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpSassignvariableop_61_adam_transformer_encoder_23_multi_head_attention_value_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpQassignvariableop_62_adam_transformer_encoder_23_multi_head_attention_value_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp^assignvariableop_63_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp\assignvariableop_64_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpKassignvariableop_65_adam_transformer_encoder_23_layer_normalization_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpJassignvariableop_66_adam_transformer_encoder_23_layer_normalization_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpMassignvariableop_67_adam_transformer_encoder_23_layer_normalization_1_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOpLassignvariableop_68_adam_transformer_encoder_23_layer_normalization_1_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp>assignvariableop_69_adam_transformer_encoder_23_dense_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp<assignvariableop_70_adam_transformer_encoder_23_dense_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp@assignvariableop_71_adam_transformer_encoder_23_dense_1_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp>assignvariableop_72_adam_transformer_encoder_23_dense_1_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpSassignvariableop_73_adam_transformer_decoder_23_multi_head_attention_query_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpQassignvariableop_74_adam_transformer_decoder_23_multi_head_attention_query_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOpQassignvariableop_75_adam_transformer_decoder_23_multi_head_attention_key_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOpOassignvariableop_76_adam_transformer_decoder_23_multi_head_attention_key_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOpSassignvariableop_77_adam_transformer_decoder_23_multi_head_attention_value_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOpQassignvariableop_78_adam_transformer_decoder_23_multi_head_attention_value_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp^assignvariableop_79_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp\assignvariableop_80_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOpKassignvariableop_81_adam_transformer_decoder_23_layer_normalization_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOpJassignvariableop_82_adam_transformer_decoder_23_layer_normalization_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOpMassignvariableop_83_adam_transformer_decoder_23_layer_normalization_1_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOpLassignvariableop_84_adam_transformer_decoder_23_layer_normalization_1_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp>assignvariableop_85_adam_transformer_decoder_23_dense_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp<assignvariableop_86_adam_transformer_decoder_23_dense_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp@assignvariableop_87_adam_transformer_decoder_23_dense_1_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp>assignvariableop_88_adam_transformer_decoder_23_dense_1_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_dense_23_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_dense_23_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_24_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_24_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOpWassignvariableop_93_adam_token_and_position_embedding_27_token_embedding28_embeddings_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOpZassignvariableop_94_adam_token_and_position_embedding_27_position_embedding28_embeddings_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOpWassignvariableop_95_adam_token_and_position_embedding_28_token_embedding29_embeddings_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOpZassignvariableop_96_adam_token_and_position_embedding_28_position_embedding29_embeddings_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOpSassignvariableop_97_adam_transformer_encoder_23_multi_head_attention_query_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0`
Identity_98IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOpQassignvariableop_98_adam_transformer_encoder_23_multi_head_attention_query_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0`
Identity_99IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOpQassignvariableop_99_adam_transformer_encoder_23_multi_head_attention_key_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOpPassignvariableop_100_adam_transformer_encoder_23_multi_head_attention_key_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOpTassignvariableop_101_adam_transformer_encoder_23_multi_head_attention_value_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOpRassignvariableop_102_adam_transformer_encoder_23_multi_head_attention_value_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp_assignvariableop_103_adam_transformer_encoder_23_multi_head_attention_attention_output_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp]assignvariableop_104_adam_transformer_encoder_23_multi_head_attention_attention_output_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOpLassignvariableop_105_adam_transformer_encoder_23_layer_normalization_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOpKassignvariableop_106_adam_transformer_encoder_23_layer_normalization_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOpNassignvariableop_107_adam_transformer_encoder_23_layer_normalization_1_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOpMassignvariableop_108_adam_transformer_encoder_23_layer_normalization_1_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp?assignvariableop_109_adam_transformer_encoder_23_dense_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp=assignvariableop_110_adam_transformer_encoder_23_dense_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOpAassignvariableop_111_adam_transformer_encoder_23_dense_1_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp?assignvariableop_112_adam_transformer_encoder_23_dense_1_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOpTassignvariableop_113_adam_transformer_decoder_23_multi_head_attention_query_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOpRassignvariableop_114_adam_transformer_decoder_23_multi_head_attention_query_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOpRassignvariableop_115_adam_transformer_decoder_23_multi_head_attention_key_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOpPassignvariableop_116_adam_transformer_decoder_23_multi_head_attention_key_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOpTassignvariableop_117_adam_transformer_decoder_23_multi_head_attention_value_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOpRassignvariableop_118_adam_transformer_decoder_23_multi_head_attention_value_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp_assignvariableop_119_adam_transformer_decoder_23_multi_head_attention_attention_output_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp]assignvariableop_120_adam_transformer_decoder_23_multi_head_attention_attention_output_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOpLassignvariableop_121_adam_transformer_decoder_23_layer_normalization_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOpKassignvariableop_122_adam_transformer_decoder_23_layer_normalization_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOpNassignvariableop_123_adam_transformer_decoder_23_layer_normalization_1_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOpMassignvariableop_124_adam_transformer_decoder_23_layer_normalization_1_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp?assignvariableop_125_adam_transformer_decoder_23_dense_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp=assignvariableop_126_adam_transformer_decoder_23_dense_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOpAassignvariableop_127_adam_transformer_decoder_23_dense_1_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp?assignvariableop_128_adam_transformer_decoder_23_dense_1_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_129Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp^StatefulPartitionedCall"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_130IdentityIdentity_129:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "%
identity_130Identity_130:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_9922
StatefulPartitionedCallStatefulPartitionedCall:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
e
F__inference_dropout_10_layer_call_and_return_conditional_losses_762631

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_761964

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_dense_23_layer_call_and_return_conditional_losses_762479

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_transformer_decoder_23_layer_call_fn_765549
decoder_sequence
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762433s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:????????? 
*
_user_specified_namedecoder_sequence
?
G
__inference__creator_766054
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_548861*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
Ω
?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_762226

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
'multi_head_attention/dropout_2/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? e
addAddV2inputsdropout/Identity:output:0*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_dense_24_layer_call_fn_766020

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_762503o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
D__inference_model_12_layer_call_and_return_conditional_losses_762510

inputs
inputs_1U
Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_13_string_lookup_13_equal_y5
1text_vectorization_13_string_lookup_13_selectv2_t	9
&token_and_position_embedding_27_762052:	?8
&token_and_position_embedding_27_762054: 8
&token_and_position_embedding_28_762086:8
&token_and_position_embedding_28_762088: 3
transformer_encoder_23_762227:/
transformer_encoder_23_762229:3
transformer_encoder_23_762231:/
transformer_encoder_23_762233:3
transformer_encoder_23_762235:/
transformer_encoder_23_762237:3
transformer_encoder_23_762239:+
transformer_encoder_23_762241:+
transformer_encoder_23_762243:+
transformer_encoder_23_762245:/
transformer_encoder_23_762247: +
transformer_encoder_23_762249: /
transformer_encoder_23_762251: +
transformer_encoder_23_762253:+
transformer_encoder_23_762255:+
transformer_encoder_23_762257:3
transformer_decoder_23_762434:/
transformer_decoder_23_762436:3
transformer_decoder_23_762438:/
transformer_decoder_23_762440:3
transformer_decoder_23_762442:/
transformer_decoder_23_762444:3
transformer_decoder_23_762446:+
transformer_decoder_23_762448:+
transformer_decoder_23_762450:+
transformer_decoder_23_762452:/
transformer_decoder_23_762454: +
transformer_decoder_23_762456: /
transformer_decoder_23_762458: +
transformer_decoder_23_762460:+
transformer_decoder_23_762462:+
transformer_decoder_23_762464:!
dense_23_762480:@
dense_23_762482:@!
dense_24_762504:@
dense_24_762506:
identity?? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall?Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?7token_and_position_embedding_27/StatefulPartitionedCall?7token_and_position_embedding_28/StatefulPartitionedCall?.transformer_decoder_23/StatefulPartitionedCall?.transformer_encoder_23/StatefulPartitionedCall~
text_vectorization_13/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????h
'text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2&text_vectorization_13/Squeeze:output:00text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
5text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/text_vectorization_13/StringSplit/strided_sliceStridedSlice9text_vectorization_13/StringSplit/StringSplitV2:indices:0>text_vectorization_13/StringSplit/strided_slice/stack:output:0@text_vectorization_13/StringSplit/strided_slice/stack_1:output:0@text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
7text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1text_vectorization_13/StringSplit/strided_slice_1StridedSlice7text_vectorization_13/StringSplit/StringSplitV2:shape:0@text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Xtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ftext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0stext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
etext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountmtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ctext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle8text_vectorization_13/StringSplit/StringSplitV2:values:0Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_13/string_lookup_13/EqualEqual8text_vectorization_13/StringSplit/StringSplitV2:values:0.text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/SelectV2SelectV20text_vectorization_13/string_lookup_13/Equal:z:01text_vectorization_13_string_lookup_13_selectv2_tMtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/IdentityIdentity8text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????t
2text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
*text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
9text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_13/RaggedToTensor/Const:output:08text_vectorization_13/string_lookup_13/Identity:output:0;text_vectorization_13/RaggedToTensor/default_value:output:0:text_vectorization_13/StringSplit/strided_slice_1:output:08text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_27/StatefulPartitionedCallStatefulPartitionedCallBtext_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_27_762052&token_and_position_embedding_27_762054*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_762051?
7token_and_position_embedding_28/StatefulPartitionedCallStatefulPartitionedCallinputs_1&token_and_position_embedding_28_762086&token_and_position_embedding_28_762088*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_762085?
add_12/PartitionedCallPartitionedCall@token_and_position_embedding_27/StatefulPartitionedCall:output:0@token_and_position_embedding_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_762097?
.transformer_encoder_23/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0transformer_encoder_23_762227transformer_encoder_23_762229transformer_encoder_23_762231transformer_encoder_23_762233transformer_encoder_23_762235transformer_encoder_23_762237transformer_encoder_23_762239transformer_encoder_23_762241transformer_encoder_23_762243transformer_encoder_23_762245transformer_encoder_23_762247transformer_encoder_23_762249transformer_encoder_23_762251transformer_encoder_23_762253transformer_encoder_23_762255transformer_encoder_23_762257*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_762226?
.transformer_decoder_23/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_23/StatefulPartitionedCall:output:0transformer_decoder_23_762434transformer_decoder_23_762436transformer_decoder_23_762438transformer_decoder_23_762440transformer_decoder_23_762442transformer_decoder_23_762444transformer_decoder_23_762446transformer_decoder_23_762448transformer_decoder_23_762450transformer_decoder_23_762452transformer_decoder_23_762454transformer_decoder_23_762456transformer_decoder_23_762458transformer_decoder_23_762460transformer_decoder_23_762462transformer_decoder_23_762464*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762433?
+global_average_pooling1d_12/PartitionedCallPartitionedCall7transformer_decoder_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_761964?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0dense_23_762480dense_23_762482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_762479?
dropout_10/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_762490?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_24_762504dense_24_762506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_762503x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCallE^text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV28^token_and_position_embedding_27/StatefulPartitionedCall8^token_and_position_embedding_28/StatefulPartitionedCall/^transformer_decoder_23/StatefulPartitionedCall/^transformer_encoder_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2?
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_27/StatefulPartitionedCall7token_and_position_embedding_27/StatefulPartitionedCall2r
7token_and_position_embedding_28/StatefulPartitionedCall7token_and_position_embedding_28/StatefulPartitionedCall2`
.transformer_decoder_23/StatefulPartitionedCall.transformer_decoder_23/StatefulPartitionedCall2`
.transformer_encoder_23/StatefulPartitionedCall.transformer_encoder_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762878
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :p
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes
: f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlicerange:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskO
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :v
range_1Rangerange_1/start:output:0strided_slice_1:output:0range_1/delta:output:0*
_output_shapes
: q
GreaterEqualGreaterEqualstrided_slice_2:output:0range_1:output:0*
T0*
_output_shapes

:  V
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0strided_slice_3:output:0strided_slice_4:output:0*
N*
T0*
_output_shapes
:a
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*"
_output_shapes
:  Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concatConcatV2ExpandDims:output:0Const:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:e
TileTileReshape:output:0concat:output:0*
T0*+
_output_shapes
:?????????  ?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acben
#multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
multi_head_attention/ExpandDims
ExpandDimsTile:output:0,multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
!multi_head_attention/softmax/CastCast(multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 multi_head_attention/softmax/subSub+multi_head_attention/softmax/sub/x:output:0%multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
 multi_head_attention/softmax/mulMul$multi_head_attention/softmax/sub:z:0+multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
 multi_head_attention/softmax/addAddV2+multi_head_attention/einsum/Einsum:output:0$multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/softmax/SoftmaxSoftmax$multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  q
,multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*multi_head_attention/dropout_2/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:05multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Cmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0z
5multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
+multi_head_attention/dropout_2/dropout/CastCast7multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/Mul_1Mul.multi_head_attention/dropout_2/dropout/Mul:z:0/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? o
addAddV2dropout/dropout/Mul_1:z:0decoder_sequence*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:] Y
+
_output_shapes
:????????? 
*
_user_specified_namedecoder_sequence
?
?
7__inference_transformer_decoder_23_layer_call_fn_765586
decoder_sequence
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_sequenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762878s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:????????? 
*
_user_specified_namedecoder_sequence
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_765999

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?E
?
__inference_adapt_step_764021
iterator9
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2~
SqueezeSqueezeIteratorGetNext:components:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2Squeeze:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:

_output_shapes
: 
??
?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765759
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :p
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes
: f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlicerange:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskO
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :v
range_1Rangerange_1/start:output:0strided_slice_1:output:0range_1/delta:output:0*
_output_shapes
: q
GreaterEqualGreaterEqualstrided_slice_2:output:0range_1:output:0*
T0*
_output_shapes

:  V
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0strided_slice_3:output:0strided_slice_4:output:0*
N*
T0*
_output_shapes
:a
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*"
_output_shapes
:  Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concatConcatV2ExpandDims:output:0Const:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:e
TileTileReshape:output:0concat:output:0*
T0*+
_output_shapes
:?????????  ?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acben
#multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
multi_head_attention/ExpandDims
ExpandDimsTile:output:0,multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
!multi_head_attention/softmax/CastCast(multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 multi_head_attention/softmax/subSub+multi_head_attention/softmax/sub/x:output:0%multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
 multi_head_attention/softmax/mulMul$multi_head_attention/softmax/sub:z:0+multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
 multi_head_attention/softmax/addAddV2+multi_head_attention/einsum/Einsum:output:0$multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/softmax/SoftmaxSoftmax$multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
'multi_head_attention/dropout_2/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? o
addAddV2dropout/Identity:output:0decoder_sequence*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:] Y
+
_output_shapes
:????????? 
*
_user_specified_namedecoder_sequence
?
s
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_765964

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?

)__inference_model_12_layer_call_fn_763587

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:

unknown_37:

unknown_38:

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallphrase
token_roleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_763402o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namephrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
token_role:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_token_and_position_embedding_27_layer_call_fn_765087

inputs	
unknown:	?
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_762051s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765953
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOpE
ShapeShapedecoder_sequence*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :p
rangeRangerange/start:output:0strided_slice_1:output:0range/delta:output:0*
_output_shapes
: f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
strided_slice_2StridedSlicerange:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskO
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :v
range_1Rangerange_1/start:output:0strided_slice_1:output:0range_1/delta:output:0*
_output_shapes
: q
GreaterEqualGreaterEqualstrided_slice_2:output:0range_1:output:0*
T0*
_output_shapes

:  V
CastCastGreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  _
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_4StridedSliceShape:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0strided_slice_3:output:0strided_slice_4:output:0*
N*
T0*
_output_shapes
:a
ReshapeReshapeCast:y:0Reshape/shape:output:0*
T0*"
_output_shapes
:  Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : {
concatConcatV2ExpandDims:output:0Const:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:e
TileTileReshape:output:0concat:output:0*
T0*+
_output_shapes
:?????????  ?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acben
#multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
multi_head_attention/ExpandDims
ExpandDimsTile:output:0,multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
!multi_head_attention/softmax/CastCast(multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 multi_head_attention/softmax/subSub+multi_head_attention/softmax/sub/x:output:0%multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  g
"multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
 multi_head_attention/softmax/mulMul$multi_head_attention/softmax/sub:z:0+multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
 multi_head_attention/softmax/addAddV2+multi_head_attention/einsum/Einsum:output:0$multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/softmax/SoftmaxSoftmax$multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  q
,multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*multi_head_attention/dropout_2/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:05multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Cmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0z
5multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
+multi_head_attention/dropout_2/dropout/CastCast7multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/Mul_1Mul.multi_head_attention/dropout_2/dropout/Mul:z:0/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? o
addAddV2dropout/dropout/Mul_1:z:0decoder_sequence*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:] Y
+
_output_shapes
:????????? 
*
_user_specified_namedecoder_sequence
?

?
D__inference_dense_24_layer_call_and_return_conditional_losses_762503

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
ݜ
?
D__inference_model_12_layer_call_and_return_conditional_losses_763871

phrase

token_roleU
Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value	2
.text_vectorization_13_string_lookup_13_equal_y5
1text_vectorization_13_string_lookup_13_selectv2_t	9
&token_and_position_embedding_27_763781:	?8
&token_and_position_embedding_27_763783: 8
&token_and_position_embedding_28_763786:8
&token_and_position_embedding_28_763788: 3
transformer_encoder_23_763792:/
transformer_encoder_23_763794:3
transformer_encoder_23_763796:/
transformer_encoder_23_763798:3
transformer_encoder_23_763800:/
transformer_encoder_23_763802:3
transformer_encoder_23_763804:+
transformer_encoder_23_763806:+
transformer_encoder_23_763808:+
transformer_encoder_23_763810:/
transformer_encoder_23_763812: +
transformer_encoder_23_763814: /
transformer_encoder_23_763816: +
transformer_encoder_23_763818:+
transformer_encoder_23_763820:+
transformer_encoder_23_763822:3
transformer_decoder_23_763825:/
transformer_decoder_23_763827:3
transformer_decoder_23_763829:/
transformer_decoder_23_763831:3
transformer_decoder_23_763833:/
transformer_decoder_23_763835:3
transformer_decoder_23_763837:+
transformer_decoder_23_763839:+
transformer_decoder_23_763841:+
transformer_decoder_23_763843:/
transformer_decoder_23_763845: +
transformer_decoder_23_763847: /
transformer_decoder_23_763849: +
transformer_decoder_23_763851:+
transformer_decoder_23_763853:+
transformer_decoder_23_763855:!
dense_23_763859:@
dense_23_763861:@!
dense_24_763865:@
dense_24_763867:
identity?? dense_23/StatefulPartitionedCall? dense_24/StatefulPartitionedCall?"dropout_10/StatefulPartitionedCall?Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2?7token_and_position_embedding_27/StatefulPartitionedCall?7token_and_position_embedding_28/StatefulPartitionedCall?.transformer_decoder_23/StatefulPartitionedCall?.transformer_encoder_23/StatefulPartitionedCall~
text_vectorization_13/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????h
'text_vectorization_13/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
/text_vectorization_13/StringSplit/StringSplitV2StringSplitV2&text_vectorization_13/Squeeze:output:00text_vectorization_13/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
5text_vectorization_13/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
7text_vectorization_13/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
7text_vectorization_13/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
/text_vectorization_13/StringSplit/strided_sliceStridedSlice9text_vectorization_13/StringSplit/StringSplitV2:indices:0>text_vectorization_13/StringSplit/strided_slice/stack:output:0@text_vectorization_13/StringSplit/strided_slice/stack_1:output:0@text_vectorization_13/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
7text_vectorization_13/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
9text_vectorization_13/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
9text_vectorization_13/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
1text_vectorization_13/StringSplit/strided_slice_1StridedSlice7text_vectorization_13/StringSplit/StringSplitV2:shape:0@text_vectorization_13/StringSplit/strided_slice_1/stack:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_1:output:0Btext_vectorization_13/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Xtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast8text_vectorization_13/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast:text_vectorization_13/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ftext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterjtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0otext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
atext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasthtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
btext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2itext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ktext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuletext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum^text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
dtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape\text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0stext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
etext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountmtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0mtext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ctext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
_text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ztext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ltext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0`text_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0htext_vectorization_13/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_table_handle8text_vectorization_13/StringSplit/StringSplitV2:values:0Rtext_vectorization_13_string_lookup_13_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
,text_vectorization_13/string_lookup_13/EqualEqual8text_vectorization_13/StringSplit/StringSplitV2:values:0.text_vectorization_13_string_lookup_13_equal_y*
T0*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/SelectV2SelectV20text_vectorization_13/string_lookup_13/Equal:z:01text_vectorization_13_string_lookup_13_selectv2_tMtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
/text_vectorization_13/string_lookup_13/IdentityIdentity8text_vectorization_13/string_lookup_13/SelectV2:output:0*
T0	*#
_output_shapes
:?????????t
2text_vectorization_13/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
*text_vectorization_13/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
9text_vectorization_13/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor3text_vectorization_13/RaggedToTensor/Const:output:08text_vectorization_13/string_lookup_13/Identity:output:0;text_vectorization_13/RaggedToTensor/default_value:output:0:text_vectorization_13/StringSplit/strided_slice_1:output:08text_vectorization_13/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_27/StatefulPartitionedCallStatefulPartitionedCallBtext_vectorization_13/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_27_763781&token_and_position_embedding_27_763783*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_762051?
7token_and_position_embedding_28/StatefulPartitionedCallStatefulPartitionedCall
token_role&token_and_position_embedding_28_763786&token_and_position_embedding_28_763788*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_762085?
add_12/PartitionedCallPartitionedCall@token_and_position_embedding_27/StatefulPartitionedCall:output:0@token_and_position_embedding_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_762097?
.transformer_encoder_23/StatefulPartitionedCallStatefulPartitionedCalladd_12/PartitionedCall:output:0transformer_encoder_23_763792transformer_encoder_23_763794transformer_encoder_23_763796transformer_encoder_23_763798transformer_encoder_23_763800transformer_encoder_23_763802transformer_encoder_23_763804transformer_encoder_23_763806transformer_encoder_23_763808transformer_encoder_23_763810transformer_encoder_23_763812transformer_encoder_23_763814transformer_encoder_23_763816transformer_encoder_23_763818transformer_encoder_23_763820transformer_encoder_23_763822*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_763101?
.transformer_decoder_23/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_23/StatefulPartitionedCall:output:0transformer_decoder_23_763825transformer_decoder_23_763827transformer_decoder_23_763829transformer_decoder_23_763831transformer_decoder_23_763833transformer_decoder_23_763835transformer_decoder_23_763837transformer_decoder_23_763839transformer_decoder_23_763841transformer_decoder_23_763843transformer_decoder_23_763845transformer_decoder_23_763847transformer_decoder_23_763849transformer_decoder_23_763851transformer_decoder_23_763853transformer_decoder_23_763855*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_762878?
+global_average_pooling1d_12/PartitionedCallPartitionedCall7transformer_decoder_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_761964?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_12/PartitionedCall:output:0dense_23_763859dense_23_763861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_762479?
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_10_layer_call_and_return_conditional_losses_762631?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_24_763865dense_24_763867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_762503x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCallE^text_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV28^token_and_position_embedding_27/StatefulPartitionedCall8^token_and_position_embedding_28/StatefulPartitionedCall/^transformer_decoder_23/StatefulPartitionedCall/^transformer_encoder_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2?
Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV2Dtext_vectorization_13/string_lookup_13/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_27/StatefulPartitionedCall7token_and_position_embedding_27/StatefulPartitionedCall2r
7token_and_position_embedding_28/StatefulPartitionedCall7token_and_position_embedding_28/StatefulPartitionedCall2`
.transformer_decoder_23/StatefulPartitionedCall.transformer_decoder_23/StatefulPartitionedCall2`
.transformer_encoder_23/StatefulPartitionedCall.transformer_encoder_23/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namephrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
token_role:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_dense_23_layer_call_fn_765973

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_762479o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ω
?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765364

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
'multi_head_attention/dropout_2/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? e
addAddV2inputsdropout/Identity:output:0*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
X
<__inference_global_average_pooling1d_12_layer_call_fn_765958

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *`
f[RY
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_761964i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
l
B__inference_add_12_layer_call_and_return_conditional_losses_762097

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:????????? S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:????????? :????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs:SO
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_763101

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource:H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:O
Amulti_head_attention_attention_output_add_readvariableop_resource:G
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:9
'dense_tensordot_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  q
,multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*multi_head_attention/dropout_2/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:05multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Cmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0z
5multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualLmulti_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0>multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
+multi_head_attention/dropout_2/dropout/CastCast7multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
,multi_head_attention/dropout_2/dropout/Mul_1Mul.multi_head_attention/dropout_2/dropout/Mul:z:0/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
$multi_head_attention/einsum_1/EinsumEinsum0multi_head_attention/dropout_2/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? e
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? |
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
 layer_normalization/moments/meanMeanadd:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
dense/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: _
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       _
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
"layer_normalization_1/moments/meanMean	add_1:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
F__inference_dropout_10_layer_call_and_return_conditional_losses_762490

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
@__inference_token_and_position_embedding_28_layer_call_fn_765123

inputs
unknown:
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_762085s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_766092
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :I
Const_1Const*
_output_shapes
: *
dtype0*
value	B :N
IdentityIdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?

)__inference_model_12_layer_call_fn_764115
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10: 

unknown_11:

unknown_12: 

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20:

unknown_21:

unknown_22: 

unknown_23:

unknown_24: 

unknown_25:

unknown_26: 

unknown_27:

unknown_28: 

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33: 

unknown_34: 

unknown_35: 

unknown_36:

unknown_37:

unknown_38:

unknown_39:@

unknown_40:@

unknown_41:@

unknown_42:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*9
Tin2
02.		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_model_12_layer_call_and_return_conditional_losses_762510o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
~:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
S
'__inference_add_12_layer_call_fn_765157
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_add_12_layer_call_and_return_conditional_losses_762097d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:????????? :????????? :U Q
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?!
?
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_765114

inputs	<
)token_embedding28_embedding_lookup_765090:	?>
,position_embedding28_readvariableop_resource: 
identity??#position_embedding28/ReadVariableOp?"token_embedding28/embedding_lookup?
"token_embedding28/embedding_lookupResourceGather)token_embedding28_embedding_lookup_765090inputs*
Tindices0	*<
_class2
0.loc:@token_embedding28/embedding_lookup/765090*+
_output_shapes
:????????? *
dtype0?
+token_embedding28/embedding_lookup/IdentityIdentity+token_embedding28/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding28/embedding_lookup/765090*+
_output_shapes
:????????? ?
-token_embedding28/embedding_lookup/Identity_1Identity4token_embedding28/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
position_embedding28/ShapeShape6token_embedding28/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding28/strided_sliceStridedSlice#position_embedding28/Shape:output:01position_embedding28/strided_slice/stack:output:03position_embedding28/strided_slice/stack_1:output:03position_embedding28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding28/ReadVariableOpReadVariableOp,position_embedding28_readvariableop_resource*
_output_shapes

: *
dtype0\
position_embedding28/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding28/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding28/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding28/strided_slice_1/stackPack#position_embedding28/Const:output:05position_embedding28/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding28/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding28/strided_slice_1/stack_1Pack+position_embedding28/strided_slice:output:07position_embedding28/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding28/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding28/strided_slice_1/stack_2Pack%position_embedding28/Const_1:output:07position_embedding28/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding28/strided_slice_1StridedSlice+position_embedding28/ReadVariableOp:value:03position_embedding28/strided_slice_1/stack:output:05position_embedding28/strided_slice_1/stack_1:output:05position_embedding28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
 position_embedding28/BroadcastToBroadcastTo-position_embedding28/strided_slice_1:output:0#position_embedding28/Shape:output:0*
T0*+
_output_shapes
:????????? ?
addAddV26token_embedding28/embedding_lookup/Identity_1:output:0)position_embedding28/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp$^position_embedding28/ReadVariableOp#^token_embedding28/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding28/ReadVariableOp#position_embedding28/ReadVariableOp2H
"token_embedding28/embedding_lookup"token_embedding28/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_transformer_encoder_23_layer_call_fn_765237

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_763101s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:????????? : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
phrase/
serving_default_phrase:0?????????
A

token_role3
serving_default_token_role:0????????? <
dense_240
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
 position_embedding"
_tf_keras_layer
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'token_embedding
(position_embedding"
_tf_keras_layer
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_multi_head_attention_layer
6_attention_layernorm
7_feedforward_layernorm
8_attention_dropout
9_intermediate_dense
:_output_dense
;_output_dropout"
_tf_keras_layer
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_self_attention_layer
 C_decoder_attention_layernorm
D_feedforward_layernorm
E_self_attention_dropout
F_intermediate_dense
G_output_dense
H_output_dropout"
_tf_keras_layer
?
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias"
_tf_keras_layer
?
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator"
_tf_keras_layer
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias"
_tf_keras_layer
?
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
U37
V38
d39
e40"
trackable_list_wrapper
?
f0
g1
h2
i3
j4
k5
l6
m7
n8
o9
p10
q11
r12
s13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
U36
V37
d38
e39"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
)__inference_model_12_layer_call_fn_762601
)__inference_model_12_layer_call_fn_764115
)__inference_model_12_layer_call_fn_764209
)__inference_model_12_layer_call_fn_763587?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
D__inference_model_12_layer_call_and_return_conditional_losses_764619
D__inference_model_12_layer_call_and_return_conditional_losses_765078
D__inference_model_12_layer_call_and_return_conditional_losses_763729
D__inference_model_12_layer_call_and_return_conditional_losses_763871?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?	capture_1
?	capture_2
?	capture_3B?
!__inference__wrapped_model_761954phrase
token_role"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateUm?Vm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Uv?Vv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
"
_generic_user_object
O
?	keras_api
?lookup_table
?token_counts"
_tf_keras_layer
?
?trace_02?
__inference_adapt_step_764021?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
@__inference_token_and_position_embedding_27_layer_call_fn_765087?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_765114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
f
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
g
embeddings
gposition_embeddings"
_tf_keras_layer
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
@__inference_token_and_position_embedding_28_layer_call_fn_765123?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_765151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
h
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
i
embeddings
iposition_embeddings"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_add_12_layer_call_fn_765157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_add_12_layer_call_and_return_conditional_losses_765163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
j0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12
w13
x14
y15"
trackable_list_wrapper
?
j0
k1
l2
m3
n4
o5
p6
q7
r8
s9
t10
u11
v12
w13
x14
y15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_transformer_encoder_23_layer_call_fn_765200
7__inference_transformer_encoder_23_layer_call_fn_765237?
???
FullArgSpec?
args7?4
jself
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults?

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765364
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765512?
???
FullArgSpec?
args7?4
jself
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults?

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	rgamma
sbeta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	tgamma
ubeta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

vkernel
wbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

xkernel
ybias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
z0
{1
|2
}3
~4
5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
?
z0
{1
|2
}3
~4
5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_transformer_decoder_23_layer_call_fn_765549
7__inference_transformer_decoder_23_layer_call_fn_765586?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
varargs
 
varkw
 #
defaults?

 

 

 

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765759
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765953?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
varargs
 
varkw
 #
defaults?

 

 

 

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_query_dense
?
_key_dense
?_value_dense
?_softmax
?_dropout_layer
?_output_dense"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
<__inference_global_average_pooling1d_12_layer_call_fn_765958?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_765964?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_23_layer_call_fn_765973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_dense_23_layer_call_and_return_conditional_losses_765984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:@2dense_23/kernel
:@2dense_23/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_10_layer_call_fn_765989
+__inference_dropout_10_layer_call_fn_765994?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_10_layer_call_and_return_conditional_losses_765999
F__inference_dropout_10_layer_call_and_return_conditional_losses_766011?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_24_layer_call_fn_766020?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_dense_24_layer_call_and_return_conditional_losses_766031?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:@2dense_24/kernel
:2dense_24/bias
O:M	?2<token_and_position_embedding_27/token_embedding28/embeddings
Q:O 2?token_and_position_embedding_27/position_embedding28/embeddings
N:L2<token_and_position_embedding_28/token_embedding29/embeddings
Q:O 2?token_and_position_embedding_28/position_embedding29/embeddings
N:L28transformer_encoder_23/multi_head_attention/query/kernel
H:F26transformer_encoder_23/multi_head_attention/query/bias
L:J26transformer_encoder_23/multi_head_attention/key/kernel
F:D24transformer_encoder_23/multi_head_attention/key/bias
N:L28transformer_encoder_23/multi_head_attention/value/kernel
H:F26transformer_encoder_23/multi_head_attention/value/bias
Y:W2Ctransformer_encoder_23/multi_head_attention/attention_output/kernel
O:M2Atransformer_encoder_23/multi_head_attention/attention_output/bias
>:<20transformer_encoder_23/layer_normalization/gamma
=:;2/transformer_encoder_23/layer_normalization/beta
@:>22transformer_encoder_23/layer_normalization_1/gamma
?:=21transformer_encoder_23/layer_normalization_1/beta
5:3 2#transformer_encoder_23/dense/kernel
/:- 2!transformer_encoder_23/dense/bias
7:5 2%transformer_encoder_23/dense_1/kernel
1:/2#transformer_encoder_23/dense_1/bias
N:L28transformer_decoder_23/multi_head_attention/query/kernel
H:F26transformer_decoder_23/multi_head_attention/query/bias
L:J26transformer_decoder_23/multi_head_attention/key/kernel
F:D24transformer_decoder_23/multi_head_attention/key/bias
N:L28transformer_decoder_23/multi_head_attention/value/kernel
H:F26transformer_decoder_23/multi_head_attention/value/bias
Y:W2Ctransformer_decoder_23/multi_head_attention/attention_output/kernel
O:M2Atransformer_decoder_23/multi_head_attention/attention_output/bias
>:<20transformer_decoder_23/layer_normalization/gamma
=:;2/transformer_decoder_23/layer_normalization/beta
@:>22transformer_decoder_23/layer_normalization_1/gamma
?:=21transformer_decoder_23/layer_normalization_1/beta
5:3 2#transformer_decoder_23/dense/kernel
/:- 2!transformer_decoder_23/dense/bias
7:5 2%transformer_decoder_23/dense_1/kernel
1:/2#transformer_decoder_23/dense_1/bias
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?	capture_1
?	capture_2
?	capture_3B?
)__inference_model_12_layer_call_fn_762601phrase
token_role"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
)__inference_model_12_layer_call_fn_764115inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
)__inference_model_12_layer_call_fn_764209inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
)__inference_model_12_layer_call_fn_763587phrase
token_role"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
D__inference_model_12_layer_call_and_return_conditional_losses_764619inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
D__inference_model_12_layer_call_and_return_conditional_losses_765078inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
D__inference_model_12_layer_call_and_return_conditional_losses_763729phrase
token_role"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
?
?	capture_1
?	capture_2
?	capture_3B?
D__inference_model_12_layer_call_and_return_conditional_losses_763871phrase
token_role"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
!J	
Const_3jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
?	capture_1
?	capture_2
?	capture_3B?
$__inference_signature_wrapper_763973phrase
token_role"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1z?	capture_2z?	capture_3
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
?
?	capture_1B?
__inference_adapt_step_764021iterator"?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?	capture_1
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
@__inference_token_and_position_embedding_27_layer_call_fn_765087inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_765114inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
f0"
trackable_list_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
g0"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
@__inference_token_and_position_embedding_28_layer_call_fn_765123inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_765151inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
h0"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
i0"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_add_12_layer_call_fn_765157inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_add_12_layer_call_and_return_conditional_losses_765163inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
Q
50
61
72
83
94
:5
;6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_transformer_encoder_23_layer_call_fn_765200inputs"?
???
FullArgSpec?
args7?4
jself
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults?

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
7__inference_transformer_encoder_23_layer_call_fn_765237inputs"?
???
FullArgSpec?
args7?4
jself
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults?

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765364inputs"?
???
FullArgSpec?
args7?4
jself
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults?

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765512inputs"?
???
FullArgSpec?
args7?4
jself
jinputs
jpadding_mask
jattention_mask
varargs
 
varkw
 
defaults?

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
X
j0
k1
l2
m3
n4
o5
p6
q7"
trackable_list_wrapper
X
j0
k1
l2
m3
n4
o5
p6
q7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecx
argsp?m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults?

 

 
p 
p 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecx
argsp?m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults?

 

 
p 
p 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

jkernel
kbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

lkernel
mbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

nkernel
obias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

pkernel
qbias"
_tf_keras_layer
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
 "
trackable_list_wrapper
Q
B0
C1
D2
E3
F4
G5
H6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_transformer_decoder_23_layer_call_fn_765549decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
varargs
 
varkw
 #
defaults?

 

 

 

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
7__inference_transformer_decoder_23_layer_call_fn_765586decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
varargs
 
varkw
 #
defaults?

 

 

 

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765759decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
varargs
 
varkw
 #
defaults?

 

 

 

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765953decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask
varargs
 
varkw
 #
defaults?

 

 

 

 

 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
Z
z0
{1
|2
}3
~4
5
?6
?7"
trackable_list_wrapper
Z
z0
{1
|2
}3
~4
5
?6
?7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpecx
argsp?m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults?

 

 
p 
p 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpecx
argsp?m
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults?

 

 
p 
p 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

zkernel
{bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

|kernel
}bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

~kernel
bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape
?kernel
	?bias"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
<__inference_global_average_pooling1d_12_layer_call_fn_765958inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_765964inputs"?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dense_23_layer_call_fn_765973inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense_23_layer_call_and_return_conditional_losses_765984inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_10_layer_call_fn_765989inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
+__inference_dropout_10_layer_call_fn_765994inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_dropout_10_layer_call_and_return_conditional_losses_765999inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_dropout_10_layer_call_and_return_conditional_losses_766011inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dense_24_layer_call_fn_766020inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense_24_layer_call_and_return_conditional_losses_766031inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_766036?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_766044?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_766049?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__creator_766054?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__initializer_766059?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference__destroyer_766064?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
!J	
Const_2jtf.TrackableConstant
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_766036"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?
?	capture_1
?	capture_2B?
__inference__initializer_766044"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?	capture_1z?	capture_2
?B?
__inference__destroyer_766049"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__creator_766054"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__initializer_766059"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference__destroyer_766064"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
&:$@2Adam/dense_23/kernel/m
 :@2Adam/dense_23/bias/m
&:$@2Adam/dense_24/kernel/m
 :2Adam/dense_24/bias/m
T:R	?2CAdam/token_and_position_embedding_27/token_embedding28/embeddings/m
V:T 2FAdam/token_and_position_embedding_27/position_embedding28/embeddings/m
S:Q2CAdam/token_and_position_embedding_28/token_embedding29/embeddings/m
V:T 2FAdam/token_and_position_embedding_28/position_embedding29/embeddings/m
S:Q2?Adam/transformer_encoder_23/multi_head_attention/query/kernel/m
M:K2=Adam/transformer_encoder_23/multi_head_attention/query/bias/m
Q:O2=Adam/transformer_encoder_23/multi_head_attention/key/kernel/m
K:I2;Adam/transformer_encoder_23/multi_head_attention/key/bias/m
S:Q2?Adam/transformer_encoder_23/multi_head_attention/value/kernel/m
M:K2=Adam/transformer_encoder_23/multi_head_attention/value/bias/m
^:\2JAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/m
T:R2HAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/m
C:A27Adam/transformer_encoder_23/layer_normalization/gamma/m
B:@26Adam/transformer_encoder_23/layer_normalization/beta/m
E:C29Adam/transformer_encoder_23/layer_normalization_1/gamma/m
D:B28Adam/transformer_encoder_23/layer_normalization_1/beta/m
::8 2*Adam/transformer_encoder_23/dense/kernel/m
4:2 2(Adam/transformer_encoder_23/dense/bias/m
<:: 2,Adam/transformer_encoder_23/dense_1/kernel/m
6:42*Adam/transformer_encoder_23/dense_1/bias/m
S:Q2?Adam/transformer_decoder_23/multi_head_attention/query/kernel/m
M:K2=Adam/transformer_decoder_23/multi_head_attention/query/bias/m
Q:O2=Adam/transformer_decoder_23/multi_head_attention/key/kernel/m
K:I2;Adam/transformer_decoder_23/multi_head_attention/key/bias/m
S:Q2?Adam/transformer_decoder_23/multi_head_attention/value/kernel/m
M:K2=Adam/transformer_decoder_23/multi_head_attention/value/bias/m
^:\2JAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/m
T:R2HAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/m
C:A27Adam/transformer_decoder_23/layer_normalization/gamma/m
B:@26Adam/transformer_decoder_23/layer_normalization/beta/m
E:C29Adam/transformer_decoder_23/layer_normalization_1/gamma/m
D:B28Adam/transformer_decoder_23/layer_normalization_1/beta/m
::8 2*Adam/transformer_decoder_23/dense/kernel/m
4:2 2(Adam/transformer_decoder_23/dense/bias/m
<:: 2,Adam/transformer_decoder_23/dense_1/kernel/m
6:42*Adam/transformer_decoder_23/dense_1/bias/m
&:$@2Adam/dense_23/kernel/v
 :@2Adam/dense_23/bias/v
&:$@2Adam/dense_24/kernel/v
 :2Adam/dense_24/bias/v
T:R	?2CAdam/token_and_position_embedding_27/token_embedding28/embeddings/v
V:T 2FAdam/token_and_position_embedding_27/position_embedding28/embeddings/v
S:Q2CAdam/token_and_position_embedding_28/token_embedding29/embeddings/v
V:T 2FAdam/token_and_position_embedding_28/position_embedding29/embeddings/v
S:Q2?Adam/transformer_encoder_23/multi_head_attention/query/kernel/v
M:K2=Adam/transformer_encoder_23/multi_head_attention/query/bias/v
Q:O2=Adam/transformer_encoder_23/multi_head_attention/key/kernel/v
K:I2;Adam/transformer_encoder_23/multi_head_attention/key/bias/v
S:Q2?Adam/transformer_encoder_23/multi_head_attention/value/kernel/v
M:K2=Adam/transformer_encoder_23/multi_head_attention/value/bias/v
^:\2JAdam/transformer_encoder_23/multi_head_attention/attention_output/kernel/v
T:R2HAdam/transformer_encoder_23/multi_head_attention/attention_output/bias/v
C:A27Adam/transformer_encoder_23/layer_normalization/gamma/v
B:@26Adam/transformer_encoder_23/layer_normalization/beta/v
E:C29Adam/transformer_encoder_23/layer_normalization_1/gamma/v
D:B28Adam/transformer_encoder_23/layer_normalization_1/beta/v
::8 2*Adam/transformer_encoder_23/dense/kernel/v
4:2 2(Adam/transformer_encoder_23/dense/bias/v
<:: 2,Adam/transformer_encoder_23/dense_1/kernel/v
6:42*Adam/transformer_encoder_23/dense_1/bias/v
S:Q2?Adam/transformer_decoder_23/multi_head_attention/query/kernel/v
M:K2=Adam/transformer_decoder_23/multi_head_attention/query/bias/v
Q:O2=Adam/transformer_decoder_23/multi_head_attention/key/kernel/v
K:I2;Adam/transformer_decoder_23/multi_head_attention/key/bias/v
S:Q2?Adam/transformer_decoder_23/multi_head_attention/value/kernel/v
M:K2=Adam/transformer_decoder_23/multi_head_attention/value/bias/v
^:\2JAdam/transformer_decoder_23/multi_head_attention/attention_output/kernel/v
T:R2HAdam/transformer_decoder_23/multi_head_attention/attention_output/bias/v
C:A27Adam/transformer_decoder_23/layer_normalization/gamma/v
B:@26Adam/transformer_decoder_23/layer_normalization/beta/v
E:C29Adam/transformer_decoder_23/layer_normalization_1/gamma/v
D:B28Adam/transformer_decoder_23/layer_normalization_1/beta/v
::8 2*Adam/transformer_decoder_23/dense/kernel/v
4:2 2(Adam/transformer_decoder_23/dense/bias/v
<:: 2,Adam/transformer_decoder_23/dense_1/kernel/v
6:42*Adam/transformer_decoder_23/dense_1/bias/v
?B?
__inference_save_fn_766083checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_766092restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	7
__inference__creator_766036?

? 
? "? 7
__inference__creator_766054?

? 
? "? 9
__inference__destroyer_766049?

? 
? "? 9
__inference__destroyer_766064?

? 
? "? C
__inference__initializer_766044 ????

? 
? "? ;
__inference__initializer_766059?

? 
? "? ?
!__inference__wrapped_model_761954?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeZ?W
P?M
K?H
 ?
phrase?????????
$?!

token_role????????? 
? "3?0
.
dense_24"?
dense_24?????????p
__inference_adapt_step_764021O??C?@
9?6
4?1?
??????????IteratorSpec 
? "
 ?
B__inference_add_12_layer_call_and_return_conditional_losses_765163?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? ")?&
?
0????????? 
? ?
'__inference_add_12_layer_call_fn_765157?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
D__inference_dense_23_layer_call_and_return_conditional_losses_765984\UV/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? |
)__inference_dense_23_layer_call_fn_765973OUV/?,
%?"
 ?
inputs?????????
? "??????????@?
D__inference_dense_24_layer_call_and_return_conditional_losses_766031\de/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? |
)__inference_dense_24_layer_call_fn_766020Ode/?,
%?"
 ?
inputs?????????@
? "???????????
F__inference_dropout_10_layer_call_and_return_conditional_losses_765999\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
F__inference_dropout_10_layer_call_and_return_conditional_losses_766011\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ~
+__inference_dropout_10_layer_call_fn_765989O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@~
+__inference_dropout_10_layer_call_fn_765994O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
W__inference_global_average_pooling1d_12_layer_call_and_return_conditional_losses_765964{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
<__inference_global_average_pooling1d_12_layer_call_fn_765958nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
D__inference_model_12_layer_call_and_return_conditional_losses_763729?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
 ?
phrase?????????
$?!

token_role????????? 
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_12_layer_call_and_return_conditional_losses_763871?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
 ?
phrase?????????
$?!

token_role????????? 
p

 
? "%?"
?
0?????????
? ?
D__inference_model_12_layer_call_and_return_conditional_losses_764619?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_12_layer_call_and_return_conditional_losses_765078?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
p

 
? "%?"
?
0?????????
? ?
)__inference_model_12_layer_call_fn_762601?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
 ?
phrase?????????
$?!

token_role????????? 
p 

 
? "???????????
)__inference_model_12_layer_call_fn_763587?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
 ?
phrase?????????
$?!

token_role????????? 
p

 
? "???????????
)__inference_model_12_layer_call_fn_764115?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
p 

 
? "???????????
)__inference_model_12_layer_call_fn_764209?:????fghijklmnopqrsvwxytuz{|}~??????????UVdeb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
p

 
? "??????????{
__inference_restore_fn_766092Z?K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_766083??&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
$__inference_signature_wrapper_763973?:????fghijklmnopqrsvwxytuz{|}~??????????UVdem?j
? 
c?`
*
phrase ?
phrase?????????
2

token_role$?!

token_role????????? "3?0
.
dense_24"?
dense_24??????????
[__inference_token_and_position_embedding_27_layer_call_and_return_conditional_losses_765114`fg/?,
%?"
 ?
inputs????????? 	
? ")?&
?
0????????? 
? ?
@__inference_token_and_position_embedding_27_layer_call_fn_765087Sfg/?,
%?"
 ?
inputs????????? 	
? "?????????? ?
[__inference_token_and_position_embedding_28_layer_call_and_return_conditional_losses_765151`hi/?,
%?"
 ?
inputs????????? 
? ")?&
?
0????????? 
? ?
@__inference_token_and_position_embedding_28_layer_call_fn_765123Shi/?,
%?"
 ?
inputs????????? 
? "?????????? ?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765759?z{|}~??????????a?^
G?D
.?+
decoder_sequence????????? 

 

 

 

 

 
?

trainingp ")?&
?
0????????? 
? ?
R__inference_transformer_decoder_23_layer_call_and_return_conditional_losses_765953?z{|}~??????????a?^
G?D
.?+
decoder_sequence????????? 

 

 

 

 

 
?

trainingp")?&
?
0????????? 
? ?
7__inference_transformer_decoder_23_layer_call_fn_765549?z{|}~??????????a?^
G?D
.?+
decoder_sequence????????? 

 

 

 

 

 
?

trainingp "?????????? ?
7__inference_transformer_decoder_23_layer_call_fn_765586?z{|}~??????????a?^
G?D
.?+
decoder_sequence????????? 

 

 

 

 

 
?

trainingp"?????????? ?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765364?jklmnopqrsvwxytuK?H
1?.
$?!
inputs????????? 

 

 
?

trainingp ")?&
?
0????????? 
? ?
R__inference_transformer_encoder_23_layer_call_and_return_conditional_losses_765512?jklmnopqrsvwxytuK?H
1?.
$?!
inputs????????? 

 

 
?

trainingp")?&
?
0????????? 
? ?
7__inference_transformer_encoder_23_layer_call_fn_765200}jklmnopqrsvwxytuK?H
1?.
$?!
inputs????????? 

 

 
?

trainingp "?????????? ?
7__inference_transformer_encoder_23_layer_call_fn_765237}jklmnopqrsvwxytuK?H
1?.
$?!
inputs????????? 

 

 
?

trainingp"?????????? 