??9
?/?/
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
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??4
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
value?!B?!?BleBpageBdeBenB
exemplaireBfoisBsurBs’BàBunBappelerBnommerB
impressionB	intitulerBl’BmoiBmerciBfichierBpourBcelaBtirageBprojetBjeBéditeBimprimerB	dénommerBpdfBimprimeBseBtaireBêtreBquiBplzBhelloBcoucouBcoolBcontenirBimageBd’BsiBpossibleBteBjusqu’Bj’BtirBtoiBmêmeB	imprimantBcopieBlanceBdocumentBtireBéditerBstpBilBnameBéditionBavanceB	souhaiterBplerB	commencerBtirerBpasBneBest-ilB	dérangerBbesoinBauraiBvouloirB
transfèreBsendBposteBenvoyerBbalanceBrapportBarticleBphotoBsortirBthanksBeffectueBaimerBm’BmeBimprimBsalutBbonjourBpleaseBaimeraiBt’B	remercierB	dénomméBsauraiBsaiBpouvoirBpeuxBpeuBfaireBfileBdessinBtyBplusBprésentationBmonBinviteB	comlimentBavecBdémarreBdémarrBcopierBattenteBplopBheyBversoBrectoBvidéoBbyeBbyBstarterBlancerB	démarrerBcodeBbalancerBsiedBmrcBdansBversionBpapierBmettreBattendreB
intitulantBfeuilleBfeuilletBzcsbdBxvtinBxrdnhBxetgBxboaegBwygwfBwbavouoBtjdxdlBsnepkBrwgiBrhldBqzvhhcBqwftxkvtBqktqnajBpyqxnsrBpteguiBpsnrBppxgB	phcgiuierBonwkfBonapklBoabfBmqfqfaBmpoaznxBlwvhgicBlvurBlthdB	ktbjiadaiBktaiBjyckBjxrrshBiqfzBipnnyuBihkayBhkqrkBgowudBgaqwqBftopafBeyebBetzlldBehwypmvoB	edyacdpakBdbtrBccwdzeaBbqkcBbcxfBayteoBhalltmsBnomméBnommantB	intituléBzrvlwBykxjBycbpBxzsmglBxzmwdBxtfaBxoiwnknzBwyzootnxBwuvcacBwungBwphnnwBwkzlqgBwkjoBwgsgkthBwevsBvujjfBvqjpBuujuBuezviBtergkBsrtbnuBsepuhBrwjvBrkikBreyymkBrbwqoBradhjnBqyqbmBqvvhnitBpxsnBpxbfwndBpttujzzBppmbzwlkBpplnkBpkdvxmBoftqfcBnshokeopBnlssovBnihnBmkuBmbgriyBlyjmohyBlrcBlibvzBlgmvBlcccwtwsBkpxybBkgjrcBkfbpbBjxeiBjqzenjBjltnhxwgBjaerbBilfvzmBijpauhBhymuBhodwBhhgvbvgBfzhorB
fvyfffdserBfghaskzuBfakBejtoBeasgBdqjvkBdemyB	damykfmunBcktkqpBbcwkBauskpBaqzsBaqorBrpwvBoiadBztpBzpsgBywhdBxlukaBxfqcqBwynduwBwjszBwjppBvqkzepBvmnfBvjgmhBuubvBubjspBswngBswednBrytwBrinfBqwtsnBqhjzuviBqgpqbBqckzvmfBplgpjBpgxojiBoyzfBoxcjgBoioawbbBndbgaiBnabfiBmvsvwBmqhgoBmoetBmmcqelBmlpaerBmfzvjBmatafreBlbxzsBlaclBkhogBjvpgBiwxztBirgbntkwBgzigdvBguuaplBgilwkogB	gbozmmdynBeehcnBdjtxbmuBdcrpBcigcBbhfsyBahwqilB	ahmecnyeeBagbgBkgogBxnxfBlqbBzzxrBzwbuBzeofByuygBwzczBwqomBvmctBvirlBroayBqbkiBpwcfBptmBnuklBmqqpBmmppBlwifBlinzBkyczBkbmhBjnwwBixscBibblBfoiBdrzpBdkeiBdaywBcunjBctzgBcskjBcdgyBzzzlBzznhBzzmddlBzzishuBzyxpmBzyvlhpBzyielBzybfBzxtdBzxndBzxicBzxgxrpzBzxdnB	zwdcuvdgpBzwcwBzvnsywBzuzfvBzuxpchhBzuwzzBzuusaoBzusfoBzuqfBzulkkBzuczBztulmiBztjwBztjbrlBzsnmhggBzsixkBzsfvqBzrrhBzqoypBzqbjrpmBzpuukBzpstBzpksBzojtqBzodamBznxcBznwizfBznpiBznatqhlqBzmxszhBzmkrBzmjyBzmjqBzmbllmBzmbgBzlxkaqBzluuuzqBzlhpBzlacbBzkwumBzkuxuBzkthBzkrxBzkmndlBzkfkfdwfBzjywstbBzjphrBziiqkBzihlBzhxosrlBzhqvgBzhaxBzgudriBzgnsqBzgkuaxtBzfaxmaBzemaBzekiBzegkajBzeemBzdxhvBzdrBzdpuBzcztzfkhBzcnpyBzcjcBzcdfBzbxaalBzbvfydnBzbtoBzbrbBzbqzBzbnoteBzbencpBzawreBzamsbByzzlByzykfByzvpiuByzgbdjByzbsfuhlByyzfdByyogyyByxxffrBywyeBywxveakBywpcztoBywgagBywdfByvygejzByvczByuyuByutioByujwBytuiBytpoBytgeoByteecpByssguBysoxpBysjjByrvjdofByrjiByqreucuBypqvwgkBypqoByplzByopwboByolzveByoffmybBynedBynarB	ymtfdrebfBymjljBymiwvBymiqzBymhshBymglBymesbbwBymbBylzmBylurnBylkqnBylggByldjqioBylddwBykwkBykosvByjxtnujByjccrByiyhzbByiktbtcByhwyByhvwyByhvfjmByhrdfByhjqBygspByghdhBygeibzByfvByflzByevdmqByeunByeujtByetqhilByetjtfaByermByekftByeipBydyyBydyBydtjwwaBydcmBycyatrBycryBycrBycnvBycgtBybtrBybnlBybmnBybdkBybalgpoByaxfubldByawwzhlByatfkByaqfyiqByalxtByaiwByahjBxzqzBxzjhBxzgglyBxymsfBxyjvxjqBxyiqyqBxyazBxyalxevlBxxkiBxxblnBxwvjBxwvdheBxwqkzBxwqfoBxwgkrBxvsqwneBxvpqhBxvdhrBxuyokBxuxpgBxuwqqfzyBxurpegBxurdmvBxuisoBxudrpBxtyqwhaBxtorBxtnlBxtdyBxtcmBxsgzgBxsbovBxrzukdmBxrytBxrswhBxrqjBxrjwBxrgkyvoBxqgpBxqfuBxphrBxpetxfBxoezgdBxnmfBxnilBxnhwBxmxvBxmwoBxmmywoeBxmjjoBxlvjaeBxlpyvzBxldwBxkzdBxkxfktBxkwkqtvBxkhooyyBxkezBxjvoollBxjsaaBxjrycqrBxjfxuvBxjdocBxjccpBxixajBxisgeiBxisfBxinnBxinlrBximkhckwBxigv
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
H
Const_3Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
*Adam/transformer_decoder_25/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/transformer_decoder_25/dense_1/bias/v
?
>Adam/transformer_decoder_25/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_25/dense_1/bias/v*
_output_shapes
: *
dtype0
?
,Adam/transformer_decoder_25/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *=
shared_name.,Adam/transformer_decoder_25/dense_1/kernel/v
?
@Adam/transformer_decoder_25/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/transformer_decoder_25/dense_1/kernel/v*
_output_shapes

:@ *
dtype0
?
(Adam/transformer_decoder_25/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/transformer_decoder_25/dense/bias/v
?
<Adam/transformer_decoder_25/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/transformer_decoder_25/dense/bias/v*
_output_shapes
:@*
dtype0
?
*Adam/transformer_decoder_25/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*;
shared_name,*Adam/transformer_decoder_25/dense/kernel/v
?
>Adam/transformer_decoder_25/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_25/dense/kernel/v*
_output_shapes

: @*
dtype0
?
8Adam/transformer_decoder_25/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_decoder_25/layer_normalization_1/beta/v
?
LAdam/transformer_decoder_25/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_25/layer_normalization_1/beta/v*
_output_shapes
: *
dtype0
?
9Adam/transformer_decoder_25/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/transformer_decoder_25/layer_normalization_1/gamma/v
?
MAdam/transformer_decoder_25/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/transformer_decoder_25/layer_normalization_1/gamma/v*
_output_shapes
: *
dtype0
?
6Adam/transformer_decoder_25/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_decoder_25/layer_normalization/beta/v
?
JAdam/transformer_decoder_25/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_25/layer_normalization/beta/v*
_output_shapes
: *
dtype0
?
7Adam/transformer_decoder_25/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_decoder_25/layer_normalization/gamma/v
?
KAdam/transformer_decoder_25/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_decoder_25/layer_normalization/gamma/v*
_output_shapes
: *
dtype0
?
HAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Y
shared_nameJHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/v
?
\Adam/transformer_decoder_25/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/v*
_output_shapes
: *
dtype0
?
JAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/v
?
^Adam/transformer_decoder_25/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/v*"
_output_shapes
: *
dtype0
?
=Adam/transformer_decoder_25/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_25/multi_head_attention/value/bias/v
?
QAdam/transformer_decoder_25/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_25/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_25/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_decoder_25/multi_head_attention/value/kernel/v
?
SAdam/transformer_decoder_25/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_25/multi_head_attention/value/kernel/v*"
_output_shapes
: *
dtype0
?
;Adam/transformer_decoder_25/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_decoder_25/multi_head_attention/key/bias/v
?
OAdam/transformer_decoder_25/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp;Adam/transformer_decoder_25/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
?
=Adam/transformer_decoder_25/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=Adam/transformer_decoder_25/multi_head_attention/key/kernel/v
?
QAdam/transformer_decoder_25/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_25/multi_head_attention/key/kernel/v*"
_output_shapes
: *
dtype0
?
=Adam/transformer_decoder_25/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_25/multi_head_attention/query/bias/v
?
QAdam/transformer_decoder_25/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_25/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_25/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_decoder_25/multi_head_attention/query/kernel/v
?
SAdam/transformer_decoder_25/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_25/multi_head_attention/query/kernel/v*"
_output_shapes
: *
dtype0
?
*Adam/transformer_encoder_25/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/transformer_encoder_25/dense_1/bias/v
?
>Adam/transformer_encoder_25/dense_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_25/dense_1/bias/v*
_output_shapes
: *
dtype0
?
,Adam/transformer_encoder_25/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *=
shared_name.,Adam/transformer_encoder_25/dense_1/kernel/v
?
@Adam/transformer_encoder_25/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/transformer_encoder_25/dense_1/kernel/v*
_output_shapes

:@ *
dtype0
?
(Adam/transformer_encoder_25/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/transformer_encoder_25/dense/bias/v
?
<Adam/transformer_encoder_25/dense/bias/v/Read/ReadVariableOpReadVariableOp(Adam/transformer_encoder_25/dense/bias/v*
_output_shapes
:@*
dtype0
?
*Adam/transformer_encoder_25/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*;
shared_name,*Adam/transformer_encoder_25/dense/kernel/v
?
>Adam/transformer_encoder_25/dense/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_25/dense/kernel/v*
_output_shapes

: @*
dtype0
?
8Adam/transformer_encoder_25/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_encoder_25/layer_normalization_1/beta/v
?
LAdam/transformer_encoder_25/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_encoder_25/layer_normalization_1/beta/v*
_output_shapes
: *
dtype0
?
9Adam/transformer_encoder_25/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/transformer_encoder_25/layer_normalization_1/gamma/v
?
MAdam/transformer_encoder_25/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp9Adam/transformer_encoder_25/layer_normalization_1/gamma/v*
_output_shapes
: *
dtype0
?
6Adam/transformer_encoder_25/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_encoder_25/layer_normalization/beta/v
?
JAdam/transformer_encoder_25/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_encoder_25/layer_normalization/beta/v*
_output_shapes
: *
dtype0
?
7Adam/transformer_encoder_25/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_encoder_25/layer_normalization/gamma/v
?
KAdam/transformer_encoder_25/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_encoder_25/layer_normalization/gamma/v*
_output_shapes
: *
dtype0
?
HAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Y
shared_nameJHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/v
?
\Adam/transformer_encoder_25/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/v*
_output_shapes
: *
dtype0
?
JAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/v
?
^Adam/transformer_encoder_25/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/v*"
_output_shapes
: *
dtype0
?
=Adam/transformer_encoder_25/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_25/multi_head_attention/value/bias/v
?
QAdam/transformer_encoder_25/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_25/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_25/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_encoder_25/multi_head_attention/value/kernel/v
?
SAdam/transformer_encoder_25/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_25/multi_head_attention/value/kernel/v*"
_output_shapes
: *
dtype0
?
;Adam/transformer_encoder_25/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_encoder_25/multi_head_attention/key/bias/v
?
OAdam/transformer_encoder_25/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp;Adam/transformer_encoder_25/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
?
=Adam/transformer_encoder_25/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=Adam/transformer_encoder_25/multi_head_attention/key/kernel/v
?
QAdam/transformer_encoder_25/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_25/multi_head_attention/key/kernel/v*"
_output_shapes
: *
dtype0
?
=Adam/transformer_encoder_25/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_25/multi_head_attention/query/bias/v
?
QAdam/transformer_encoder_25/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_25/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_25/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_encoder_25/multi_head_attention/query/kernel/v
?
SAdam/transformer_encoder_25/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_25/multi_head_attention/query/kernel/v*"
_output_shapes
: *
dtype0
?
FAdam/token_and_position_embedding_24/position_embedding25/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/token_and_position_embedding_24/position_embedding25/embeddings/v
?
ZAdam/token_and_position_embedding_24/position_embedding25/embeddings/v/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_24/position_embedding25/embeddings/v*
_output_shapes

:  *
dtype0
?
CAdam/token_and_position_embedding_24/token_embedding25/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *T
shared_nameECAdam/token_and_position_embedding_24/token_embedding25/embeddings/v
?
WAdam/token_and_position_embedding_24/token_embedding25/embeddings/v/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_24/token_embedding25/embeddings/v*
_output_shapes

: *
dtype0
?
FAdam/token_and_position_embedding_23/position_embedding24/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/token_and_position_embedding_23/position_embedding24/embeddings/v
?
ZAdam/token_and_position_embedding_23/position_embedding24/embeddings/v/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_23/position_embedding24/embeddings/v*
_output_shapes

:  *
dtype0
?
CAdam/token_and_position_embedding_23/token_embedding24/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *T
shared_nameECAdam/token_and_position_embedding_23/token_embedding24/embeddings/v
?
WAdam/token_and_position_embedding_23/token_embedding24/embeddings/v/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_23/token_embedding24/embeddings/v*
_output_shapes
:	? *
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes

:  *
dtype0
?
*Adam/transformer_decoder_25/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/transformer_decoder_25/dense_1/bias/m
?
>Adam/transformer_decoder_25/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_25/dense_1/bias/m*
_output_shapes
: *
dtype0
?
,Adam/transformer_decoder_25/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *=
shared_name.,Adam/transformer_decoder_25/dense_1/kernel/m
?
@Adam/transformer_decoder_25/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/transformer_decoder_25/dense_1/kernel/m*
_output_shapes

:@ *
dtype0
?
(Adam/transformer_decoder_25/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/transformer_decoder_25/dense/bias/m
?
<Adam/transformer_decoder_25/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/transformer_decoder_25/dense/bias/m*
_output_shapes
:@*
dtype0
?
*Adam/transformer_decoder_25/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*;
shared_name,*Adam/transformer_decoder_25/dense/kernel/m
?
>Adam/transformer_decoder_25/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_decoder_25/dense/kernel/m*
_output_shapes

: @*
dtype0
?
8Adam/transformer_decoder_25/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_decoder_25/layer_normalization_1/beta/m
?
LAdam/transformer_decoder_25/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_25/layer_normalization_1/beta/m*
_output_shapes
: *
dtype0
?
9Adam/transformer_decoder_25/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/transformer_decoder_25/layer_normalization_1/gamma/m
?
MAdam/transformer_decoder_25/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/transformer_decoder_25/layer_normalization_1/gamma/m*
_output_shapes
: *
dtype0
?
6Adam/transformer_decoder_25/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_decoder_25/layer_normalization/beta/m
?
JAdam/transformer_decoder_25/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_25/layer_normalization/beta/m*
_output_shapes
: *
dtype0
?
7Adam/transformer_decoder_25/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_decoder_25/layer_normalization/gamma/m
?
KAdam/transformer_decoder_25/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_decoder_25/layer_normalization/gamma/m*
_output_shapes
: *
dtype0
?
HAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Y
shared_nameJHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/m
?
\Adam/transformer_decoder_25/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/m*
_output_shapes
: *
dtype0
?
JAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/m
?
^Adam/transformer_decoder_25/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/m*"
_output_shapes
: *
dtype0
?
=Adam/transformer_decoder_25/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_25/multi_head_attention/value/bias/m
?
QAdam/transformer_decoder_25/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_25/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_25/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_decoder_25/multi_head_attention/value/kernel/m
?
SAdam/transformer_decoder_25/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_25/multi_head_attention/value/kernel/m*"
_output_shapes
: *
dtype0
?
;Adam/transformer_decoder_25/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_decoder_25/multi_head_attention/key/bias/m
?
OAdam/transformer_decoder_25/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp;Adam/transformer_decoder_25/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
?
=Adam/transformer_decoder_25/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=Adam/transformer_decoder_25/multi_head_attention/key/kernel/m
?
QAdam/transformer_decoder_25/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_25/multi_head_attention/key/kernel/m*"
_output_shapes
: *
dtype0
?
=Adam/transformer_decoder_25/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_decoder_25/multi_head_attention/query/bias/m
?
QAdam/transformer_decoder_25/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_decoder_25/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_decoder_25/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_decoder_25/multi_head_attention/query/kernel/m
?
SAdam/transformer_decoder_25/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_decoder_25/multi_head_attention/query/kernel/m*"
_output_shapes
: *
dtype0
?
*Adam/transformer_encoder_25/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/transformer_encoder_25/dense_1/bias/m
?
>Adam/transformer_encoder_25/dense_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_25/dense_1/bias/m*
_output_shapes
: *
dtype0
?
,Adam/transformer_encoder_25/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *=
shared_name.,Adam/transformer_encoder_25/dense_1/kernel/m
?
@Adam/transformer_encoder_25/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/transformer_encoder_25/dense_1/kernel/m*
_output_shapes

:@ *
dtype0
?
(Adam/transformer_encoder_25/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(Adam/transformer_encoder_25/dense/bias/m
?
<Adam/transformer_encoder_25/dense/bias/m/Read/ReadVariableOpReadVariableOp(Adam/transformer_encoder_25/dense/bias/m*
_output_shapes
:@*
dtype0
?
*Adam/transformer_encoder_25/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*;
shared_name,*Adam/transformer_encoder_25/dense/kernel/m
?
>Adam/transformer_encoder_25/dense/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/transformer_encoder_25/dense/kernel/m*
_output_shapes

: @*
dtype0
?
8Adam/transformer_encoder_25/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_encoder_25/layer_normalization_1/beta/m
?
LAdam/transformer_encoder_25/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_encoder_25/layer_normalization_1/beta/m*
_output_shapes
: *
dtype0
?
9Adam/transformer_encoder_25/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/transformer_encoder_25/layer_normalization_1/gamma/m
?
MAdam/transformer_encoder_25/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp9Adam/transformer_encoder_25/layer_normalization_1/gamma/m*
_output_shapes
: *
dtype0
?
6Adam/transformer_encoder_25/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_encoder_25/layer_normalization/beta/m
?
JAdam/transformer_encoder_25/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_encoder_25/layer_normalization/beta/m*
_output_shapes
: *
dtype0
?
7Adam/transformer_encoder_25/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_encoder_25/layer_normalization/gamma/m
?
KAdam/transformer_encoder_25/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_encoder_25/layer_normalization/gamma/m*
_output_shapes
: *
dtype0
?
HAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Y
shared_nameJHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/m
?
\Adam/transformer_encoder_25/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/m*
_output_shapes
: *
dtype0
?
JAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *[
shared_nameLJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/m
?
^Adam/transformer_encoder_25/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/m*"
_output_shapes
: *
dtype0
?
=Adam/transformer_encoder_25/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_25/multi_head_attention/value/bias/m
?
QAdam/transformer_encoder_25/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_25/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_25/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_encoder_25/multi_head_attention/value/kernel/m
?
SAdam/transformer_encoder_25/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_25/multi_head_attention/value/kernel/m*"
_output_shapes
: *
dtype0
?
;Adam/transformer_encoder_25/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*L
shared_name=;Adam/transformer_encoder_25/multi_head_attention/key/bias/m
?
OAdam/transformer_encoder_25/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp;Adam/transformer_encoder_25/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
?
=Adam/transformer_encoder_25/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=Adam/transformer_encoder_25/multi_head_attention/key/kernel/m
?
QAdam/transformer_encoder_25/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_25/multi_head_attention/key/kernel/m*"
_output_shapes
: *
dtype0
?
=Adam/transformer_encoder_25/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/transformer_encoder_25/multi_head_attention/query/bias/m
?
QAdam/transformer_encoder_25/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_encoder_25/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
?
?Adam/transformer_encoder_25/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?Adam/transformer_encoder_25/multi_head_attention/query/kernel/m
?
SAdam/transformer_encoder_25/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_encoder_25/multi_head_attention/query/kernel/m*"
_output_shapes
: *
dtype0
?
FAdam/token_and_position_embedding_24/position_embedding25/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/token_and_position_embedding_24/position_embedding25/embeddings/m
?
ZAdam/token_and_position_embedding_24/position_embedding25/embeddings/m/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_24/position_embedding25/embeddings/m*
_output_shapes

:  *
dtype0
?
CAdam/token_and_position_embedding_24/token_embedding25/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *T
shared_nameECAdam/token_and_position_embedding_24/token_embedding25/embeddings/m
?
WAdam/token_and_position_embedding_24/token_embedding25/embeddings/m/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_24/token_embedding25/embeddings/m*
_output_shapes

: *
dtype0
?
FAdam/token_and_position_embedding_23/position_embedding24/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/token_and_position_embedding_23/position_embedding24/embeddings/m
?
ZAdam/token_and_position_embedding_23/position_embedding24/embeddings/m/Read/ReadVariableOpReadVariableOpFAdam/token_and_position_embedding_23/position_embedding24/embeddings/m*
_output_shapes

:  *
dtype0
?
CAdam/token_and_position_embedding_23/token_embedding24/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *T
shared_nameECAdam/token_and_position_embedding_23/token_embedding24/embeddings/m
?
WAdam/token_and_position_embedding_23/token_embedding24/embeddings/m/Read/ReadVariableOpReadVariableOpCAdam/token_and_position_embedding_23/token_embedding24/embeddings/m*
_output_shapes
:	? *
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes

:  *
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
shared_nametable_340481*
value_dtype0	
n

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name393026*
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
#transformer_decoder_25/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#transformer_decoder_25/dense_1/bias
?
7transformer_decoder_25/dense_1/bias/Read/ReadVariableOpReadVariableOp#transformer_decoder_25/dense_1/bias*
_output_shapes
: *
dtype0
?
%transformer_decoder_25/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *6
shared_name'%transformer_decoder_25/dense_1/kernel
?
9transformer_decoder_25/dense_1/kernel/Read/ReadVariableOpReadVariableOp%transformer_decoder_25/dense_1/kernel*
_output_shapes

:@ *
dtype0
?
!transformer_decoder_25/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!transformer_decoder_25/dense/bias
?
5transformer_decoder_25/dense/bias/Read/ReadVariableOpReadVariableOp!transformer_decoder_25/dense/bias*
_output_shapes
:@*
dtype0
?
#transformer_decoder_25/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*4
shared_name%#transformer_decoder_25/dense/kernel
?
7transformer_decoder_25/dense/kernel/Read/ReadVariableOpReadVariableOp#transformer_decoder_25/dense/kernel*
_output_shapes

: @*
dtype0
?
1transformer_decoder_25/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31transformer_decoder_25/layer_normalization_1/beta
?
Etransformer_decoder_25/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp1transformer_decoder_25/layer_normalization_1/beta*
_output_shapes
: *
dtype0
?
2transformer_decoder_25/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42transformer_decoder_25/layer_normalization_1/gamma
?
Ftransformer_decoder_25/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp2transformer_decoder_25/layer_normalization_1/gamma*
_output_shapes
: *
dtype0
?
/transformer_decoder_25/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_decoder_25/layer_normalization/beta
?
Ctransformer_decoder_25/layer_normalization/beta/Read/ReadVariableOpReadVariableOp/transformer_decoder_25/layer_normalization/beta*
_output_shapes
: *
dtype0
?
0transformer_decoder_25/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_decoder_25/layer_normalization/gamma
?
Dtransformer_decoder_25/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp0transformer_decoder_25/layer_normalization/gamma*
_output_shapes
: *
dtype0
?
Atransformer_decoder_25/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *R
shared_nameCAtransformer_decoder_25/multi_head_attention/attention_output/bias
?
Utransformer_decoder_25/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpAtransformer_decoder_25/multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
?
Ctransformer_decoder_25/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECtransformer_decoder_25/multi_head_attention/attention_output/kernel
?
Wtransformer_decoder_25/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpCtransformer_decoder_25/multi_head_attention/attention_output/kernel*"
_output_shapes
: *
dtype0
?
6transformer_decoder_25/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_decoder_25/multi_head_attention/value/bias
?
Jtransformer_decoder_25/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp6transformer_decoder_25/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
?
8transformer_decoder_25/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8transformer_decoder_25/multi_head_attention/value/kernel
?
Ltransformer_decoder_25/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp8transformer_decoder_25/multi_head_attention/value/kernel*"
_output_shapes
: *
dtype0
?
4transformer_decoder_25/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64transformer_decoder_25/multi_head_attention/key/bias
?
Htransformer_decoder_25/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp4transformer_decoder_25/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
?
6transformer_decoder_25/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86transformer_decoder_25/multi_head_attention/key/kernel
?
Jtransformer_decoder_25/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp6transformer_decoder_25/multi_head_attention/key/kernel*"
_output_shapes
: *
dtype0
?
6transformer_decoder_25/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_decoder_25/multi_head_attention/query/bias
?
Jtransformer_decoder_25/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp6transformer_decoder_25/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
?
8transformer_decoder_25/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8transformer_decoder_25/multi_head_attention/query/kernel
?
Ltransformer_decoder_25/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp8transformer_decoder_25/multi_head_attention/query/kernel*"
_output_shapes
: *
dtype0
?
#transformer_encoder_25/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#transformer_encoder_25/dense_1/bias
?
7transformer_encoder_25/dense_1/bias/Read/ReadVariableOpReadVariableOp#transformer_encoder_25/dense_1/bias*
_output_shapes
: *
dtype0
?
%transformer_encoder_25/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *6
shared_name'%transformer_encoder_25/dense_1/kernel
?
9transformer_encoder_25/dense_1/kernel/Read/ReadVariableOpReadVariableOp%transformer_encoder_25/dense_1/kernel*
_output_shapes

:@ *
dtype0
?
!transformer_encoder_25/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!transformer_encoder_25/dense/bias
?
5transformer_encoder_25/dense/bias/Read/ReadVariableOpReadVariableOp!transformer_encoder_25/dense/bias*
_output_shapes
:@*
dtype0
?
#transformer_encoder_25/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*4
shared_name%#transformer_encoder_25/dense/kernel
?
7transformer_encoder_25/dense/kernel/Read/ReadVariableOpReadVariableOp#transformer_encoder_25/dense/kernel*
_output_shapes

: @*
dtype0
?
1transformer_encoder_25/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31transformer_encoder_25/layer_normalization_1/beta
?
Etransformer_encoder_25/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp1transformer_encoder_25/layer_normalization_1/beta*
_output_shapes
: *
dtype0
?
2transformer_encoder_25/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42transformer_encoder_25/layer_normalization_1/gamma
?
Ftransformer_encoder_25/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp2transformer_encoder_25/layer_normalization_1/gamma*
_output_shapes
: *
dtype0
?
/transformer_encoder_25/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_encoder_25/layer_normalization/beta
?
Ctransformer_encoder_25/layer_normalization/beta/Read/ReadVariableOpReadVariableOp/transformer_encoder_25/layer_normalization/beta*
_output_shapes
: *
dtype0
?
0transformer_encoder_25/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_encoder_25/layer_normalization/gamma
?
Dtransformer_encoder_25/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp0transformer_encoder_25/layer_normalization/gamma*
_output_shapes
: *
dtype0
?
Atransformer_encoder_25/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *R
shared_nameCAtransformer_encoder_25/multi_head_attention/attention_output/bias
?
Utransformer_encoder_25/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOpAtransformer_encoder_25/multi_head_attention/attention_output/bias*
_output_shapes
: *
dtype0
?
Ctransformer_encoder_25/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECtransformer_encoder_25/multi_head_attention/attention_output/kernel
?
Wtransformer_encoder_25/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpCtransformer_encoder_25/multi_head_attention/attention_output/kernel*"
_output_shapes
: *
dtype0
?
6transformer_encoder_25/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_encoder_25/multi_head_attention/value/bias
?
Jtransformer_encoder_25/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp6transformer_encoder_25/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
?
8transformer_encoder_25/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8transformer_encoder_25/multi_head_attention/value/kernel
?
Ltransformer_encoder_25/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp8transformer_encoder_25/multi_head_attention/value/kernel*"
_output_shapes
: *
dtype0
?
4transformer_encoder_25/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*E
shared_name64transformer_encoder_25/multi_head_attention/key/bias
?
Htransformer_encoder_25/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp4transformer_encoder_25/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
?
6transformer_encoder_25/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86transformer_encoder_25/multi_head_attention/key/kernel
?
Jtransformer_encoder_25/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp6transformer_encoder_25/multi_head_attention/key/kernel*"
_output_shapes
: *
dtype0
?
6transformer_encoder_25/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86transformer_encoder_25/multi_head_attention/query/bias
?
Jtransformer_encoder_25/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp6transformer_encoder_25/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
?
8transformer_encoder_25/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8transformer_encoder_25/multi_head_attention/query/kernel
?
Ltransformer_encoder_25/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp8transformer_encoder_25/multi_head_attention/query/kernel*"
_output_shapes
: *
dtype0
?
?token_and_position_embedding_24/position_embedding25/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?token_and_position_embedding_24/position_embedding25/embeddings
?
Stoken_and_position_embedding_24/position_embedding25/embeddings/Read/ReadVariableOpReadVariableOp?token_and_position_embedding_24/position_embedding25/embeddings*
_output_shapes

:  *
dtype0
?
<token_and_position_embedding_24/token_embedding25/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><token_and_position_embedding_24/token_embedding25/embeddings
?
Ptoken_and_position_embedding_24/token_embedding25/embeddings/Read/ReadVariableOpReadVariableOp<token_and_position_embedding_24/token_embedding25/embeddings*
_output_shapes

: *
dtype0
?
?token_and_position_embedding_23/position_embedding24/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?token_and_position_embedding_23/position_embedding24/embeddings
?
Stoken_and_position_embedding_23/position_embedding24/embeddings/Read/ReadVariableOpReadVariableOp?token_and_position_embedding_23/position_embedding24/embeddings*
_output_shapes

:  *
dtype0
?
<token_and_position_embedding_23/token_embedding24/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *M
shared_name><token_and_position_embedding_23/token_embedding24/embeddings
?
Ptoken_and_position_embedding_23/token_embedding24/embeddings/Read/ReadVariableOpReadVariableOp<token_and_position_embedding_23/token_embedding24/embeddings*
_output_shapes
:	? *
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
: *
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:  *
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
hash_tableConst_4Const_3Const_5<token_and_position_embedding_23/token_embedding24/embeddings?token_and_position_embedding_23/position_embedding24/embeddings<token_and_position_embedding_24/token_embedding25/embeddings?token_and_position_embedding_24/position_embedding25/embeddings8transformer_encoder_25/multi_head_attention/query/kernel6transformer_encoder_25/multi_head_attention/query/bias6transformer_encoder_25/multi_head_attention/key/kernel4transformer_encoder_25/multi_head_attention/key/bias8transformer_encoder_25/multi_head_attention/value/kernel6transformer_encoder_25/multi_head_attention/value/biasCtransformer_encoder_25/multi_head_attention/attention_output/kernelAtransformer_encoder_25/multi_head_attention/attention_output/bias0transformer_encoder_25/layer_normalization/gamma/transformer_encoder_25/layer_normalization/beta#transformer_encoder_25/dense/kernel!transformer_encoder_25/dense/bias%transformer_encoder_25/dense_1/kernel#transformer_encoder_25/dense_1/bias2transformer_encoder_25/layer_normalization_1/gamma1transformer_encoder_25/layer_normalization_1/beta8transformer_decoder_25/multi_head_attention/query/kernel6transformer_decoder_25/multi_head_attention/query/bias6transformer_decoder_25/multi_head_attention/key/kernel4transformer_decoder_25/multi_head_attention/key/bias8transformer_decoder_25/multi_head_attention/value/kernel6transformer_decoder_25/multi_head_attention/value/biasCtransformer_decoder_25/multi_head_attention/attention_output/kernelAtransformer_decoder_25/multi_head_attention/attention_output/bias0transformer_decoder_25/layer_normalization/gamma/transformer_decoder_25/layer_normalization/beta#transformer_decoder_25/dense/kernel!transformer_decoder_25/dense/bias%transformer_decoder_25/dense_1/kernel#transformer_decoder_25/dense_1/bias2transformer_decoder_25/layer_normalization_1/gamma1transformer_decoder_25/layer_normalization_1/betadense_14/kerneldense_14/bias*7
Tin0
.2,		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *H
_read_only_resource_inputs*
(&	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_502587
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
__inference__initializer_504580
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
__inference__initializer_504595
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
?
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
;
	keras_api
_lookup_layer
_adapt_function*
* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding*
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%token_embedding
&position_embedding*
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_multi_head_attention_layer
4_attention_layernorm
5_feedforward_layernorm
6_attention_dropout
7_intermediate_dense
8_output_dense
9_output_dropout*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_self_attention_layer
 A_decoder_attention_layernorm
B_feedforward_layernorm
C_self_attention_dropout
D_intermediate_dense
E_output_dense
F_output_dropout*
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias*
?
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13
b14
c15
d16
e17
f18
g19
h20
i21
j22
k23
l24
m25
n26
o27
p28
q29
r30
s31
t32
u33
v34
w35
x36
S37
T38*
?
U0
V1
W2
X3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
g18
h19
i20
j21
k22
l23
m24
n25
o26
p27
q28
r29
s30
t31
u32
v33
w34
x35
S36
T37*
* 
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
8
~trace_0
trace_1
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
?learning_rateSm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?*
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
U0
V1*

U0
V1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
U
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
V
embeddings
Vposition_embeddings*

W0
X1*

W0
X1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
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
W
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
X
embeddings
Xposition_embeddings*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
z
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15*
z
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

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
	agamma
bbeta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	cgamma
dbeta*
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
+?&call_and_return_all_conditional_losses

ekernel
fbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

gkernel
hbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
z
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15*
z
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

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
	qgamma
rbeta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	sgamma
tbeta*
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

ukernel
vbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

wkernel
xbias*
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
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

S0
T1*

S0
T1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<token_and_position_embedding_23/token_embedding24/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?token_and_position_embedding_23/position_embedding24/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE<token_and_position_embedding_24/token_embedding25/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?token_and_position_embedding_24/position_embedding25/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_encoder_25/multi_head_attention/query/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_encoder_25/multi_head_attention/query/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_encoder_25/multi_head_attention/key/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4transformer_encoder_25/multi_head_attention/key/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_encoder_25/multi_head_attention/value/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_encoder_25/multi_head_attention/value/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUECtransformer_encoder_25/multi_head_attention/attention_output/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAtransformer_encoder_25/multi_head_attention/attention_output/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_encoder_25/layer_normalization/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_encoder_25/layer_normalization/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2transformer_encoder_25/layer_normalization_1/gamma'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_encoder_25/layer_normalization_1/beta'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_encoder_25/dense/kernel'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_encoder_25/dense/bias'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%transformer_encoder_25/dense_1/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_encoder_25/dense_1/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_decoder_25/multi_head_attention/query/kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_decoder_25/multi_head_attention/query/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_decoder_25/multi_head_attention/key/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4transformer_decoder_25/multi_head_attention/key/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE8transformer_decoder_25/multi_head_attention/value/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6transformer_decoder_25/multi_head_attention/value/bias'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUECtransformer_decoder_25/multi_head_attention/attention_output/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAtransformer_decoder_25/multi_head_attention/attention_output/bias'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_decoder_25/layer_normalization/gamma'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_decoder_25/layer_normalization/beta'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2transformer_decoder_25/layer_normalization_1/gamma'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_decoder_25/layer_normalization_1/beta'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_decoder_25/dense/kernel'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!transformer_decoder_25/dense/bias'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%transformer_decoder_25/dense_1/kernel'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#transformer_decoder_25/dense_1/bias'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
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
9*
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
0
1*
* 
* 
* 
* 
* 

U0*

U0*
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

V0*

V0*
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
%0
&1*
* 
* 
* 
* 
* 

W0*

W0*
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

X0*

X0*
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
* 
* 
* 
* 
* 
* 
* 
5
30
41
52
63
74
85
96*
* 
* 
* 
* 
* 
* 
* 
<
Y0
Z1
[2
\3
]4
^5
_6
`7*
<
Y0
Z1
[2
\3
]4
^5
_6
`7*
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
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

Ykernel
Zbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

[kernel
\bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

]kernel
^bias*
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

_kernel
`bias*

a0
b1*

a0
b1*
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
c0
d1*

c0
d1*
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
e0
f1*

e0
f1*
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

g0
h1*

g0
h1*
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
* 
5
@0
A1
B2
C3
D4
E5
F6*
* 
* 
* 
* 
* 
* 
* 
<
i0
j1
k2
l3
m4
n5
o6
p7*
<
i0
j1
k2
l3
m4
n5
o6
p7*
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

ikernel
jbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

kkernel
lbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

mkernel
nbias*
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

okernel
pbias*

q0
r1*

q0
r1*
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

s0
t1*

s0
t1*
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

u0
v1*

u0
v1*
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

w0
x1*

w0
x1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
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
Y0
Z1*

Y0
Z1*
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

[0
\1*

[0
\1*
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

]0
^1*

]0
^1*
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
_0
`1*

_0
`1*
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
i0
j1*

i0
j1*
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
k0
l1*

k0
l1*
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
m0
n1*

m0
n1*
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
o0
p1*

o0
p1*
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
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_23/token_embedding24/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_23/position_embedding24/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_24/token_embedding25/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_24/position_embedding25/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_25/multi_head_attention/query/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_25/multi_head_attention/query/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_25/multi_head_attention/key/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_encoder_25/multi_head_attention/key/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_25/multi_head_attention/value/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_25/multi_head_attention/value/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_encoder_25/layer_normalization/gamma/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_encoder_25/layer_normalization/beta/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_encoder_25/layer_normalization_1/gamma/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_encoder_25/layer_normalization_1/beta/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_25/dense/kernel/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_encoder_25/dense/bias/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_encoder_25/dense_1/kernel/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_25/dense_1/bias/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_25/multi_head_attention/query/kernel/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_25/multi_head_attention/query/bias/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_25/multi_head_attention/key/kernel/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_decoder_25/multi_head_attention/key/bias/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_25/multi_head_attention/value/kernel/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_25/multi_head_attention/value/bias/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_decoder_25/layer_normalization/gamma/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_decoder_25/layer_normalization/beta/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_decoder_25/layer_normalization_1/gamma/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_decoder_25/layer_normalization_1/beta/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_25/dense/kernel/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_decoder_25/dense/bias/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_decoder_25/dense_1/kernel/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_25/dense_1/bias/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_23/token_embedding24/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_23/position_embedding24/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUECAdam/token_and_position_embedding_24/token_embedding25/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/token_and_position_embedding_24/position_embedding25/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_25/multi_head_attention/query/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_25/multi_head_attention/query/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_25/multi_head_attention/key/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_encoder_25/multi_head_attention/key/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_encoder_25/multi_head_attention/value/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_encoder_25/multi_head_attention/value/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_encoder_25/layer_normalization/gamma/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_encoder_25/layer_normalization/beta/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_encoder_25/layer_normalization_1/gamma/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_encoder_25/layer_normalization_1/beta/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_25/dense/kernel/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_encoder_25/dense/bias/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_encoder_25/dense_1/kernel/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_encoder_25/dense_1/bias/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_25/multi_head_attention/query/kernel/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_25/multi_head_attention/query/bias/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_25/multi_head_attention/key/kernel/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE;Adam/transformer_decoder_25/multi_head_attention/key/bias/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE?Adam/transformer_decoder_25/multi_head_attention/value/kernel/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=Adam/transformer_decoder_25/multi_head_attention/value/bias/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_decoder_25/layer_normalization/gamma/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_decoder_25/layer_normalization/beta/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE9Adam/transformer_decoder_25/layer_normalization_1/gamma/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_decoder_25/layer_normalization_1/beta/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_25/dense/kernel/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE(Adam/transformer_decoder_25/dense/bias/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/transformer_decoder_25/dense_1/kernel/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/transformer_decoder_25/dense_1/bias/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?I
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOpPtoken_and_position_embedding_23/token_embedding24/embeddings/Read/ReadVariableOpStoken_and_position_embedding_23/position_embedding24/embeddings/Read/ReadVariableOpPtoken_and_position_embedding_24/token_embedding25/embeddings/Read/ReadVariableOpStoken_and_position_embedding_24/position_embedding25/embeddings/Read/ReadVariableOpLtransformer_encoder_25/multi_head_attention/query/kernel/Read/ReadVariableOpJtransformer_encoder_25/multi_head_attention/query/bias/Read/ReadVariableOpJtransformer_encoder_25/multi_head_attention/key/kernel/Read/ReadVariableOpHtransformer_encoder_25/multi_head_attention/key/bias/Read/ReadVariableOpLtransformer_encoder_25/multi_head_attention/value/kernel/Read/ReadVariableOpJtransformer_encoder_25/multi_head_attention/value/bias/Read/ReadVariableOpWtransformer_encoder_25/multi_head_attention/attention_output/kernel/Read/ReadVariableOpUtransformer_encoder_25/multi_head_attention/attention_output/bias/Read/ReadVariableOpDtransformer_encoder_25/layer_normalization/gamma/Read/ReadVariableOpCtransformer_encoder_25/layer_normalization/beta/Read/ReadVariableOpFtransformer_encoder_25/layer_normalization_1/gamma/Read/ReadVariableOpEtransformer_encoder_25/layer_normalization_1/beta/Read/ReadVariableOp7transformer_encoder_25/dense/kernel/Read/ReadVariableOp5transformer_encoder_25/dense/bias/Read/ReadVariableOp9transformer_encoder_25/dense_1/kernel/Read/ReadVariableOp7transformer_encoder_25/dense_1/bias/Read/ReadVariableOpLtransformer_decoder_25/multi_head_attention/query/kernel/Read/ReadVariableOpJtransformer_decoder_25/multi_head_attention/query/bias/Read/ReadVariableOpJtransformer_decoder_25/multi_head_attention/key/kernel/Read/ReadVariableOpHtransformer_decoder_25/multi_head_attention/key/bias/Read/ReadVariableOpLtransformer_decoder_25/multi_head_attention/value/kernel/Read/ReadVariableOpJtransformer_decoder_25/multi_head_attention/value/bias/Read/ReadVariableOpWtransformer_decoder_25/multi_head_attention/attention_output/kernel/Read/ReadVariableOpUtransformer_decoder_25/multi_head_attention/attention_output/bias/Read/ReadVariableOpDtransformer_decoder_25/layer_normalization/gamma/Read/ReadVariableOpCtransformer_decoder_25/layer_normalization/beta/Read/ReadVariableOpFtransformer_decoder_25/layer_normalization_1/gamma/Read/ReadVariableOpEtransformer_decoder_25/layer_normalization_1/beta/Read/ReadVariableOp7transformer_decoder_25/dense/kernel/Read/ReadVariableOp5transformer_decoder_25/dense/bias/Read/ReadVariableOp9transformer_decoder_25/dense_1/kernel/Read/ReadVariableOp7transformer_decoder_25/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOpWAdam/token_and_position_embedding_23/token_embedding24/embeddings/m/Read/ReadVariableOpZAdam/token_and_position_embedding_23/position_embedding24/embeddings/m/Read/ReadVariableOpWAdam/token_and_position_embedding_24/token_embedding25/embeddings/m/Read/ReadVariableOpZAdam/token_and_position_embedding_24/position_embedding25/embeddings/m/Read/ReadVariableOpSAdam/transformer_encoder_25/multi_head_attention/query/kernel/m/Read/ReadVariableOpQAdam/transformer_encoder_25/multi_head_attention/query/bias/m/Read/ReadVariableOpQAdam/transformer_encoder_25/multi_head_attention/key/kernel/m/Read/ReadVariableOpOAdam/transformer_encoder_25/multi_head_attention/key/bias/m/Read/ReadVariableOpSAdam/transformer_encoder_25/multi_head_attention/value/kernel/m/Read/ReadVariableOpQAdam/transformer_encoder_25/multi_head_attention/value/bias/m/Read/ReadVariableOp^Adam/transformer_encoder_25/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOp\Adam/transformer_encoder_25/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpKAdam/transformer_encoder_25/layer_normalization/gamma/m/Read/ReadVariableOpJAdam/transformer_encoder_25/layer_normalization/beta/m/Read/ReadVariableOpMAdam/transformer_encoder_25/layer_normalization_1/gamma/m/Read/ReadVariableOpLAdam/transformer_encoder_25/layer_normalization_1/beta/m/Read/ReadVariableOp>Adam/transformer_encoder_25/dense/kernel/m/Read/ReadVariableOp<Adam/transformer_encoder_25/dense/bias/m/Read/ReadVariableOp@Adam/transformer_encoder_25/dense_1/kernel/m/Read/ReadVariableOp>Adam/transformer_encoder_25/dense_1/bias/m/Read/ReadVariableOpSAdam/transformer_decoder_25/multi_head_attention/query/kernel/m/Read/ReadVariableOpQAdam/transformer_decoder_25/multi_head_attention/query/bias/m/Read/ReadVariableOpQAdam/transformer_decoder_25/multi_head_attention/key/kernel/m/Read/ReadVariableOpOAdam/transformer_decoder_25/multi_head_attention/key/bias/m/Read/ReadVariableOpSAdam/transformer_decoder_25/multi_head_attention/value/kernel/m/Read/ReadVariableOpQAdam/transformer_decoder_25/multi_head_attention/value/bias/m/Read/ReadVariableOp^Adam/transformer_decoder_25/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOp\Adam/transformer_decoder_25/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpKAdam/transformer_decoder_25/layer_normalization/gamma/m/Read/ReadVariableOpJAdam/transformer_decoder_25/layer_normalization/beta/m/Read/ReadVariableOpMAdam/transformer_decoder_25/layer_normalization_1/gamma/m/Read/ReadVariableOpLAdam/transformer_decoder_25/layer_normalization_1/beta/m/Read/ReadVariableOp>Adam/transformer_decoder_25/dense/kernel/m/Read/ReadVariableOp<Adam/transformer_decoder_25/dense/bias/m/Read/ReadVariableOp@Adam/transformer_decoder_25/dense_1/kernel/m/Read/ReadVariableOp>Adam/transformer_decoder_25/dense_1/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOpWAdam/token_and_position_embedding_23/token_embedding24/embeddings/v/Read/ReadVariableOpZAdam/token_and_position_embedding_23/position_embedding24/embeddings/v/Read/ReadVariableOpWAdam/token_and_position_embedding_24/token_embedding25/embeddings/v/Read/ReadVariableOpZAdam/token_and_position_embedding_24/position_embedding25/embeddings/v/Read/ReadVariableOpSAdam/transformer_encoder_25/multi_head_attention/query/kernel/v/Read/ReadVariableOpQAdam/transformer_encoder_25/multi_head_attention/query/bias/v/Read/ReadVariableOpQAdam/transformer_encoder_25/multi_head_attention/key/kernel/v/Read/ReadVariableOpOAdam/transformer_encoder_25/multi_head_attention/key/bias/v/Read/ReadVariableOpSAdam/transformer_encoder_25/multi_head_attention/value/kernel/v/Read/ReadVariableOpQAdam/transformer_encoder_25/multi_head_attention/value/bias/v/Read/ReadVariableOp^Adam/transformer_encoder_25/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOp\Adam/transformer_encoder_25/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpKAdam/transformer_encoder_25/layer_normalization/gamma/v/Read/ReadVariableOpJAdam/transformer_encoder_25/layer_normalization/beta/v/Read/ReadVariableOpMAdam/transformer_encoder_25/layer_normalization_1/gamma/v/Read/ReadVariableOpLAdam/transformer_encoder_25/layer_normalization_1/beta/v/Read/ReadVariableOp>Adam/transformer_encoder_25/dense/kernel/v/Read/ReadVariableOp<Adam/transformer_encoder_25/dense/bias/v/Read/ReadVariableOp@Adam/transformer_encoder_25/dense_1/kernel/v/Read/ReadVariableOp>Adam/transformer_encoder_25/dense_1/bias/v/Read/ReadVariableOpSAdam/transformer_decoder_25/multi_head_attention/query/kernel/v/Read/ReadVariableOpQAdam/transformer_decoder_25/multi_head_attention/query/bias/v/Read/ReadVariableOpQAdam/transformer_decoder_25/multi_head_attention/key/kernel/v/Read/ReadVariableOpOAdam/transformer_decoder_25/multi_head_attention/key/bias/v/Read/ReadVariableOpSAdam/transformer_decoder_25/multi_head_attention/value/kernel/v/Read/ReadVariableOpQAdam/transformer_decoder_25/multi_head_attention/value/bias/v/Read/ReadVariableOp^Adam/transformer_decoder_25/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOp\Adam/transformer_decoder_25/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpKAdam/transformer_decoder_25/layer_normalization/gamma/v/Read/ReadVariableOpJAdam/transformer_decoder_25/layer_normalization/beta/v/Read/ReadVariableOpMAdam/transformer_decoder_25/layer_normalization_1/gamma/v/Read/ReadVariableOpLAdam/transformer_decoder_25/layer_normalization_1/beta/v/Read/ReadVariableOp>Adam/transformer_decoder_25/dense/kernel/v/Read/ReadVariableOp<Adam/transformer_decoder_25/dense/bias/v/Read/ReadVariableOp@Adam/transformer_decoder_25/dense_1/kernel/v/Read/ReadVariableOp>Adam/transformer_decoder_25/dense_1/bias/v/Read/ReadVariableOpConst_6*?
Tin?
?2		*
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
__inference__traced_save_505035
?5
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenamedense_14/kerneldense_14/bias<token_and_position_embedding_23/token_embedding24/embeddings?token_and_position_embedding_23/position_embedding24/embeddings<token_and_position_embedding_24/token_embedding25/embeddings?token_and_position_embedding_24/position_embedding25/embeddings8transformer_encoder_25/multi_head_attention/query/kernel6transformer_encoder_25/multi_head_attention/query/bias6transformer_encoder_25/multi_head_attention/key/kernel4transformer_encoder_25/multi_head_attention/key/bias8transformer_encoder_25/multi_head_attention/value/kernel6transformer_encoder_25/multi_head_attention/value/biasCtransformer_encoder_25/multi_head_attention/attention_output/kernelAtransformer_encoder_25/multi_head_attention/attention_output/bias0transformer_encoder_25/layer_normalization/gamma/transformer_encoder_25/layer_normalization/beta2transformer_encoder_25/layer_normalization_1/gamma1transformer_encoder_25/layer_normalization_1/beta#transformer_encoder_25/dense/kernel!transformer_encoder_25/dense/bias%transformer_encoder_25/dense_1/kernel#transformer_encoder_25/dense_1/bias8transformer_decoder_25/multi_head_attention/query/kernel6transformer_decoder_25/multi_head_attention/query/bias6transformer_decoder_25/multi_head_attention/key/kernel4transformer_decoder_25/multi_head_attention/key/bias8transformer_decoder_25/multi_head_attention/value/kernel6transformer_decoder_25/multi_head_attention/value/biasCtransformer_decoder_25/multi_head_attention/attention_output/kernelAtransformer_decoder_25/multi_head_attention/attention_output/bias0transformer_decoder_25/layer_normalization/gamma/transformer_decoder_25/layer_normalization/beta2transformer_decoder_25/layer_normalization_1/gamma1transformer_decoder_25/layer_normalization_1/beta#transformer_decoder_25/dense/kernel!transformer_decoder_25/dense/bias%transformer_decoder_25/dense_1/kernel#transformer_decoder_25/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/dense_14/kernel/mAdam/dense_14/bias/mCAdam/token_and_position_embedding_23/token_embedding24/embeddings/mFAdam/token_and_position_embedding_23/position_embedding24/embeddings/mCAdam/token_and_position_embedding_24/token_embedding25/embeddings/mFAdam/token_and_position_embedding_24/position_embedding25/embeddings/m?Adam/transformer_encoder_25/multi_head_attention/query/kernel/m=Adam/transformer_encoder_25/multi_head_attention/query/bias/m=Adam/transformer_encoder_25/multi_head_attention/key/kernel/m;Adam/transformer_encoder_25/multi_head_attention/key/bias/m?Adam/transformer_encoder_25/multi_head_attention/value/kernel/m=Adam/transformer_encoder_25/multi_head_attention/value/bias/mJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/mHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/m7Adam/transformer_encoder_25/layer_normalization/gamma/m6Adam/transformer_encoder_25/layer_normalization/beta/m9Adam/transformer_encoder_25/layer_normalization_1/gamma/m8Adam/transformer_encoder_25/layer_normalization_1/beta/m*Adam/transformer_encoder_25/dense/kernel/m(Adam/transformer_encoder_25/dense/bias/m,Adam/transformer_encoder_25/dense_1/kernel/m*Adam/transformer_encoder_25/dense_1/bias/m?Adam/transformer_decoder_25/multi_head_attention/query/kernel/m=Adam/transformer_decoder_25/multi_head_attention/query/bias/m=Adam/transformer_decoder_25/multi_head_attention/key/kernel/m;Adam/transformer_decoder_25/multi_head_attention/key/bias/m?Adam/transformer_decoder_25/multi_head_attention/value/kernel/m=Adam/transformer_decoder_25/multi_head_attention/value/bias/mJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/mHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/m7Adam/transformer_decoder_25/layer_normalization/gamma/m6Adam/transformer_decoder_25/layer_normalization/beta/m9Adam/transformer_decoder_25/layer_normalization_1/gamma/m8Adam/transformer_decoder_25/layer_normalization_1/beta/m*Adam/transformer_decoder_25/dense/kernel/m(Adam/transformer_decoder_25/dense/bias/m,Adam/transformer_decoder_25/dense_1/kernel/m*Adam/transformer_decoder_25/dense_1/bias/mAdam/dense_14/kernel/vAdam/dense_14/bias/vCAdam/token_and_position_embedding_23/token_embedding24/embeddings/vFAdam/token_and_position_embedding_23/position_embedding24/embeddings/vCAdam/token_and_position_embedding_24/token_embedding25/embeddings/vFAdam/token_and_position_embedding_24/position_embedding25/embeddings/v?Adam/transformer_encoder_25/multi_head_attention/query/kernel/v=Adam/transformer_encoder_25/multi_head_attention/query/bias/v=Adam/transformer_encoder_25/multi_head_attention/key/kernel/v;Adam/transformer_encoder_25/multi_head_attention/key/bias/v?Adam/transformer_encoder_25/multi_head_attention/value/kernel/v=Adam/transformer_encoder_25/multi_head_attention/value/bias/vJAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/vHAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/v7Adam/transformer_encoder_25/layer_normalization/gamma/v6Adam/transformer_encoder_25/layer_normalization/beta/v9Adam/transformer_encoder_25/layer_normalization_1/gamma/v8Adam/transformer_encoder_25/layer_normalization_1/beta/v*Adam/transformer_encoder_25/dense/kernel/v(Adam/transformer_encoder_25/dense/bias/v,Adam/transformer_encoder_25/dense_1/kernel/v*Adam/transformer_encoder_25/dense_1/bias/v?Adam/transformer_decoder_25/multi_head_attention/query/kernel/v=Adam/transformer_decoder_25/multi_head_attention/query/bias/v=Adam/transformer_decoder_25/multi_head_attention/key/kernel/v;Adam/transformer_decoder_25/multi_head_attention/key/bias/v?Adam/transformer_decoder_25/multi_head_attention/value/kernel/v=Adam/transformer_decoder_25/multi_head_attention/value/bias/vJAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/vHAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/v7Adam/transformer_decoder_25/layer_normalization/gamma/v6Adam/transformer_decoder_25/layer_normalization/beta/v9Adam/transformer_decoder_25/layer_normalization_1/gamma/v8Adam/transformer_decoder_25/layer_normalization_1/beta/v*Adam/transformer_decoder_25/dense/kernel/v(Adam/transformer_decoder_25/dense/bias/v,Adam/transformer_decoder_25/dense_1/kernel/v*Adam/transformer_decoder_25/dense_1/bias/v*?
Tin?
2}*
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
"__inference__traced_restore_505426??-
?
?	
(__inference_model_9_layer_call_fn_502815
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	? 
	unknown_4:  
	unknown_5: 
	unknown_6:  
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22:  

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39:  

unknown_40: 
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
unknown_40*7
Tin0
.2,		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *H
_read_only_resource_inputs*
(&	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_9_layer_call_and_return_conditional_losses_502040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
?
m
A__inference_add_9_layer_call_and_return_conditional_losses_503746
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:?????????  S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????  :?????????  :U Q
+
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
?
R
&__inference_add_9_layer_call_fn_503740
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
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_500806d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????  :?????????  :U Q
+
_output_shapes
:?????????  
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????  
"
_user_specified_name
inputs/1
?E
?
__inference_adapt_step_502635
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
Ω
?
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_503947

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  e
addAddV2inputsdropout/Identity:output:0*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
 
_user_specified_nameinputs
?
-
__inference__destroyer_504585
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
?
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501142
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
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
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  o
addAddV2dropout/Identity:output:0decoder_sequence*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
*
_user_specified_namedecoder_sequence
?
?
)__inference_dense_14_layer_call_fn_504556

inputs
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_501188o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
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
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504536
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
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
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  o
addAddV2dropout/dropout/Mul_1:z:0decoder_sequence*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
*
_user_specified_namedecoder_sequence
Ω
?
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_500935

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  e
addAddV2inputsdropout/Identity:output:0*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
 
_user_specified_nameinputs
?
?
7__inference_transformer_encoder_25_layer_call_fn_503820

inputs
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: @

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 
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
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_501749s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
D__inference_dense_14_layer_call_and_return_conditional_losses_501188

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:????????? `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?	
(__inference_model_9_layer_call_fn_501282

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	? 
	unknown_4:  
	unknown_5: 
	unknown_6:  
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22:  

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39:  

unknown_40: 
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
unknown_40*7
Tin0
.2,		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *H
_read_only_resource_inputs*
(&	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_9_layer_call_and_return_conditional_losses_501195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
?	
?
__inference_restore_fn_504628
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
??
?
C__inference_model_9_layer_call_and_return_conditional_losses_502489

phrase

token_roleS
Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_9_string_lookup_9_equal_y3
/text_vectorization_9_string_lookup_9_selectv2_t	9
&token_and_position_embedding_23_502405:	? 8
&token_and_position_embedding_23_502407:  8
&token_and_position_embedding_24_502410: 8
&token_and_position_embedding_24_502412:  3
transformer_encoder_25_502416: /
transformer_encoder_25_502418:3
transformer_encoder_25_502420: /
transformer_encoder_25_502422:3
transformer_encoder_25_502424: /
transformer_encoder_25_502426:3
transformer_encoder_25_502428: +
transformer_encoder_25_502430: +
transformer_encoder_25_502432: +
transformer_encoder_25_502434: /
transformer_encoder_25_502436: @+
transformer_encoder_25_502438:@/
transformer_encoder_25_502440:@ +
transformer_encoder_25_502442: +
transformer_encoder_25_502444: +
transformer_encoder_25_502446: 3
transformer_decoder_25_502449: /
transformer_decoder_25_502451:3
transformer_decoder_25_502453: /
transformer_decoder_25_502455:3
transformer_decoder_25_502457: /
transformer_decoder_25_502459:3
transformer_decoder_25_502461: +
transformer_decoder_25_502463: +
transformer_decoder_25_502465: +
transformer_decoder_25_502467: /
transformer_decoder_25_502469: @+
transformer_decoder_25_502471:@/
transformer_decoder_25_502473:@ +
transformer_decoder_25_502475: +
transformer_decoder_25_502477: +
transformer_decoder_25_502479: !
dense_14_502483:  
dense_14_502485: 
identity?? dense_14/StatefulPartitionedCall?Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?7token_and_position_embedding_23/StatefulPartitionedCall?7token_and_position_embedding_24/StatefulPartitionedCall?.transformer_decoder_25/StatefulPartitionedCall?.transformer_encoder_25/StatefulPartitionedCall}
text_vectorization_9/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_9/StringSplit/StringSplitV2StringSplitV2%text_vectorization_9/Squeeze:output:0/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_9/StringSplit/strided_sliceStridedSlice8text_vectorization_9/StringSplit/StringSplitV2:indices:0=text_vectorization_9/StringSplit/strided_slice/stack:output:0?text_vectorization_9/StringSplit/strided_slice/stack_1:output:0?text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_9/StringSplit/strided_slice_1StridedSlice6text_vectorization_9/StringSplit/StringSplitV2:shape:0?text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
itext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
dtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_9/StringSplit/StringSplitV2:values:0Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_9/string_lookup_9/EqualEqual7text_vectorization_9/StringSplit/StringSplitV2:values:0,text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/SelectV2SelectV2.text_vectorization_9/string_lookup_9/Equal:z:0/text_vectorization_9_string_lookup_9_selectv2_tKtext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/IdentityIdentity6text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
8text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_9/RaggedToTensor/Const:output:06text_vectorization_9/string_lookup_9/Identity:output:0:text_vectorization_9/RaggedToTensor/default_value:output:09text_vectorization_9/StringSplit/strided_slice_1:output:07text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_23/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_23_502405&token_and_position_embedding_23_502407*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_500760?
7token_and_position_embedding_24/StatefulPartitionedCallStatefulPartitionedCall
token_role&token_and_position_embedding_24_502410&token_and_position_embedding_24_502412*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_500794?
add_9/PartitionedCallPartitionedCall@token_and_position_embedding_23/StatefulPartitionedCall:output:0@token_and_position_embedding_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_500806?
.transformer_encoder_25/StatefulPartitionedCallStatefulPartitionedCalladd_9/PartitionedCall:output:0transformer_encoder_25_502416transformer_encoder_25_502418transformer_encoder_25_502420transformer_encoder_25_502422transformer_encoder_25_502424transformer_encoder_25_502426transformer_encoder_25_502428transformer_encoder_25_502430transformer_encoder_25_502432transformer_encoder_25_502434transformer_encoder_25_502436transformer_encoder_25_502438transformer_encoder_25_502440transformer_encoder_25_502442transformer_encoder_25_502444transformer_encoder_25_502446*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_501749?
.transformer_decoder_25/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_25/StatefulPartitionedCall:output:0transformer_decoder_25_502449transformer_decoder_25_502451transformer_decoder_25_502453transformer_decoder_25_502455transformer_decoder_25_502457transformer_decoder_25_502459transformer_decoder_25_502461transformer_decoder_25_502463transformer_decoder_25_502465transformer_decoder_25_502467transformer_decoder_25_502469transformer_decoder_25_502471transformer_decoder_25_502473transformer_decoder_25_502475transformer_decoder_25_502477transformer_decoder_25_502479*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501526?
*global_average_pooling1d_9/PartitionedCallPartitionedCall7transformer_decoder_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_500673?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_9/PartitionedCall:output:0dense_14_502483dense_14_502485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_501188x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_14/StatefulPartitionedCallC^text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV28^token_and_position_embedding_23/StatefulPartitionedCall8^token_and_position_embedding_24/StatefulPartitionedCall/^transformer_decoder_25/StatefulPartitionedCall/^transformer_encoder_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_23/StatefulPartitionedCall7token_and_position_embedding_23/StatefulPartitionedCall2r
7token_and_position_embedding_24/StatefulPartitionedCall7token_and_position_embedding_24/StatefulPartitionedCall2`
.transformer_decoder_25/StatefulPartitionedCall.transformer_decoder_25/StatefulPartitionedCall2`
.transformer_encoder_25/StatefulPartitionedCall.transformer_encoder_25/StatefulPartitionedCall:O K
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
??
?4
C__inference_model_9_layer_call_and_return_conditional_losses_503217
inputs_0
inputs_1S
Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_9_string_lookup_9_equal_y3
/text_vectorization_9_string_lookup_9_selectv2_t	\
Itoken_and_position_embedding_23_token_embedding24_embedding_lookup_502867:	? ^
Ltoken_and_position_embedding_23_position_embedding24_readvariableop_resource:  [
Itoken_and_position_embedding_24_token_embedding25_embedding_lookup_502891: ^
Ltoken_and_position_embedding_24_position_embedding25_readvariableop_resource:  m
Wtransformer_encoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource: _
Mtransformer_encoder_25_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_encoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource: ]
Ktransformer_encoder_25_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_encoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource: _
Mtransformer_encoder_25_multi_head_attention_value_add_readvariableop_resource:x
btransformer_encoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: f
Xtransformer_encoder_25_multi_head_attention_attention_output_add_readvariableop_resource: ^
Ptransformer_encoder_25_layer_normalization_batchnorm_mul_readvariableop_resource: Z
Ltransformer_encoder_25_layer_normalization_batchnorm_readvariableop_resource: P
>transformer_encoder_25_dense_tensordot_readvariableop_resource: @J
<transformer_encoder_25_dense_biasadd_readvariableop_resource:@R
@transformer_encoder_25_dense_1_tensordot_readvariableop_resource:@ L
>transformer_encoder_25_dense_1_biasadd_readvariableop_resource: `
Rtransformer_encoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource: \
Ntransformer_encoder_25_layer_normalization_1_batchnorm_readvariableop_resource: m
Wtransformer_decoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource: _
Mtransformer_decoder_25_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_decoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource: ]
Ktransformer_decoder_25_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_decoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource: _
Mtransformer_decoder_25_multi_head_attention_value_add_readvariableop_resource:x
btransformer_decoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: f
Xtransformer_decoder_25_multi_head_attention_attention_output_add_readvariableop_resource: ^
Ptransformer_decoder_25_layer_normalization_batchnorm_mul_readvariableop_resource: Z
Ltransformer_decoder_25_layer_normalization_batchnorm_readvariableop_resource: P
>transformer_decoder_25_dense_tensordot_readvariableop_resource: @J
<transformer_decoder_25_dense_biasadd_readvariableop_resource:@R
@transformer_decoder_25_dense_1_tensordot_readvariableop_resource:@ L
>transformer_decoder_25_dense_1_biasadd_readvariableop_resource: `
Rtransformer_decoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource: \
Ntransformer_decoder_25_layer_normalization_1_batchnorm_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource:  6
(dense_14_biasadd_readvariableop_resource: 
identity??dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?Ctoken_and_position_embedding_23/position_embedding24/ReadVariableOp?Btoken_and_position_embedding_23/token_embedding24/embedding_lookup?Ctoken_and_position_embedding_24/position_embedding25/ReadVariableOp?Btoken_and_position_embedding_24/token_embedding25/embedding_lookup?3transformer_decoder_25/dense/BiasAdd/ReadVariableOp?5transformer_decoder_25/dense/Tensordot/ReadVariableOp?5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp?7transformer_decoder_25/dense_1/Tensordot/ReadVariableOp?Ctransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_decoder_25/multi_head_attention/key/add/ReadVariableOp?Ltransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOp?Ntransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOp?Ntransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp?3transformer_encoder_25/dense/BiasAdd/ReadVariableOp?5transformer_encoder_25/dense/Tensordot/ReadVariableOp?5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp?7transformer_encoder_25/dense_1/Tensordot/ReadVariableOp?Ctransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_encoder_25/multi_head_attention/key/add/ReadVariableOp?Ltransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOp?Ntransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOp?Ntransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp
text_vectorization_9/SqueezeSqueezeinputs_0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_9/StringSplit/StringSplitV2StringSplitV2%text_vectorization_9/Squeeze:output:0/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_9/StringSplit/strided_sliceStridedSlice8text_vectorization_9/StringSplit/StringSplitV2:indices:0=text_vectorization_9/StringSplit/strided_slice/stack:output:0?text_vectorization_9/StringSplit/strided_slice/stack_1:output:0?text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_9/StringSplit/strided_slice_1StridedSlice6text_vectorization_9/StringSplit/StringSplitV2:shape:0?text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
itext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
dtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_9/StringSplit/StringSplitV2:values:0Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_9/string_lookup_9/EqualEqual7text_vectorization_9/StringSplit/StringSplitV2:values:0,text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/SelectV2SelectV2.text_vectorization_9/string_lookup_9/Equal:z:0/text_vectorization_9_string_lookup_9_selectv2_tKtext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/IdentityIdentity6text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
8text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_9/RaggedToTensor/Const:output:06text_vectorization_9/string_lookup_9/Identity:output:0:text_vectorization_9/RaggedToTensor/default_value:output:09text_vectorization_9/StringSplit/strided_slice_1:output:07text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Btoken_and_position_embedding_23/token_embedding24/embedding_lookupResourceGatherItoken_and_position_embedding_23_token_embedding24_embedding_lookup_502867Atext_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*\
_classR
PNloc:@token_and_position_embedding_23/token_embedding24/embedding_lookup/502867*+
_output_shapes
:?????????  *
dtype0?
Ktoken_and_position_embedding_23/token_embedding24/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_23/token_embedding24/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_23/token_embedding24/embedding_lookup/502867*+
_output_shapes
:?????????  ?
Mtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
:token_and_position_embedding_23/position_embedding24/ShapeShapeVtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_23/position_embedding24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_23/position_embedding24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_23/position_embedding24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_23/position_embedding24/strided_sliceStridedSliceCtoken_and_position_embedding_23/position_embedding24/Shape:output:0Qtoken_and_position_embedding_23/position_embedding24/strided_slice/stack:output:0Stoken_and_position_embedding_23/position_embedding24/strided_slice/stack_1:output:0Stoken_and_position_embedding_23/position_embedding24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_23/position_embedding24/ReadVariableOpReadVariableOpLtoken_and_position_embedding_23_position_embedding24_readvariableop_resource*
_output_shapes

:  *
dtype0|
:token_and_position_embedding_23/position_embedding24/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_23/position_embedding24/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_23/position_embedding24/strided_slice_1/stackPackCtoken_and_position_embedding_23/position_embedding24/Const:output:0Utoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1PackKtoken_and_position_embedding_23/position_embedding24/strided_slice:output:0Wtoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2PackEtoken_and_position_embedding_23/position_embedding24/Const_1:output:0Wtoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_23/position_embedding24/strided_slice_1StridedSliceKtoken_and_position_embedding_23/position_embedding24/ReadVariableOp:value:0Stoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack:output:0Utoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
@token_and_position_embedding_23/position_embedding24/BroadcastToBroadcastToMtoken_and_position_embedding_23/position_embedding24/strided_slice_1:output:0Ctoken_and_position_embedding_23/position_embedding24/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
#token_and_position_embedding_23/addAddV2Vtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_23/position_embedding24/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  ?
6token_and_position_embedding_24/token_embedding25/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
Btoken_and_position_embedding_24/token_embedding25/embedding_lookupResourceGatherItoken_and_position_embedding_24_token_embedding25_embedding_lookup_502891:token_and_position_embedding_24/token_embedding25/Cast:y:0*
Tindices0*\
_classR
PNloc:@token_and_position_embedding_24/token_embedding25/embedding_lookup/502891*+
_output_shapes
:?????????  *
dtype0?
Ktoken_and_position_embedding_24/token_embedding25/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_24/token_embedding25/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_24/token_embedding25/embedding_lookup/502891*+
_output_shapes
:?????????  ?
Mtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
:token_and_position_embedding_24/position_embedding25/ShapeShapeVtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_24/position_embedding25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_24/position_embedding25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_24/position_embedding25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_24/position_embedding25/strided_sliceStridedSliceCtoken_and_position_embedding_24/position_embedding25/Shape:output:0Qtoken_and_position_embedding_24/position_embedding25/strided_slice/stack:output:0Stoken_and_position_embedding_24/position_embedding25/strided_slice/stack_1:output:0Stoken_and_position_embedding_24/position_embedding25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_24/position_embedding25/ReadVariableOpReadVariableOpLtoken_and_position_embedding_24_position_embedding25_readvariableop_resource*
_output_shapes

:  *
dtype0|
:token_and_position_embedding_24/position_embedding25/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_24/position_embedding25/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_24/position_embedding25/strided_slice_1/stackPackCtoken_and_position_embedding_24/position_embedding25/Const:output:0Utoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1PackKtoken_and_position_embedding_24/position_embedding25/strided_slice:output:0Wtoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2PackEtoken_and_position_embedding_24/position_embedding25/Const_1:output:0Wtoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_24/position_embedding25/strided_slice_1StridedSliceKtoken_and_position_embedding_24/position_embedding25/ReadVariableOp:value:0Stoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack:output:0Utoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
@token_and_position_embedding_24/position_embedding25/BroadcastToBroadcastToMtoken_and_position_embedding_24/position_embedding25/strided_slice_1:output:0Ctoken_and_position_embedding_24/position_embedding25/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
#token_and_position_embedding_24/addAddV2Vtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_24/position_embedding25/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  ?
	add_9/addAddV2'token_and_position_embedding_23/add:z:0'token_and_position_embedding_24/add:z:0*
T0*+
_output_shapes
:?????????  ?
Ntransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_encoder_25/multi_head_attention/query/einsum/EinsumEinsumadd_9/add:z:0Vtransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_encoder_25_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_25/multi_head_attention/query/addAddV2Htransformer_encoder_25/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_encoder_25/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_encoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
=transformer_encoder_25/multi_head_attention/key/einsum/EinsumEinsumadd_9/add:z:0Ttransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_encoder_25/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_encoder_25_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_encoder_25/multi_head_attention/key/addAddV2Ftransformer_encoder_25/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_encoder_25/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_encoder_25/multi_head_attention/value/einsum/EinsumEinsumadd_9/add:z:0Vtransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_encoder_25_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_25/multi_head_attention/value/addAddV2Htransformer_encoder_25/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_encoder_25/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_encoder_25/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
/transformer_encoder_25/multi_head_attention/MulMul9transformer_encoder_25/multi_head_attention/query/add:z:0:transformer_encoder_25/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_encoder_25/multi_head_attention/einsum/EinsumEinsum7transformer_encoder_25/multi_head_attention/key/add:z:03transformer_encoder_25/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
;transformer_encoder_25/multi_head_attention/softmax/SoftmaxSoftmaxBtransformer_encoder_25/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
>transformer_encoder_25/multi_head_attention/dropout_2/IdentityIdentityEtransformer_encoder_25/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
;transformer_encoder_25/multi_head_attention/einsum_1/EinsumEinsumGtransformer_encoder_25/multi_head_attention/dropout_2/Identity:output:09transformer_encoder_25/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_encoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Jtransformer_encoder_25/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_encoder_25/multi_head_attention/einsum_1/Einsum:output:0atransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
Otransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_encoder_25_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
@transformer_encoder_25/multi_head_attention/attention_output/addAddV2Stransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
'transformer_encoder_25/dropout/IdentityIdentityDtransformer_encoder_25/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  ?
transformer_encoder_25/addAddV2add_9/add:z:00transformer_encoder_25/dropout/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Itransformer_encoder_25/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_encoder_25/layer_normalization/moments/meanMeantransformer_encoder_25/add:z:0Rtransformer_encoder_25/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_encoder_25/layer_normalization/moments/StopGradientStopGradient@transformer_encoder_25/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_encoder_25/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_25/add:z:0Htransformer_encoder_25/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Mtransformer_encoder_25/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_encoder_25/layer_normalization/moments/varianceMeanHtransformer_encoder_25/layer_normalization/moments/SquaredDifference:z:0Vtransformer_encoder_25/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_encoder_25/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_encoder_25/layer_normalization/batchnorm/addAddV2Dtransformer_encoder_25/layer_normalization/moments/variance:output:0Ctransformer_encoder_25/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_25/layer_normalization/batchnorm/RsqrtRsqrt<transformer_encoder_25/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_encoder_25_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_encoder_25/layer_normalization/batchnorm/mulMul>transformer_encoder_25/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
:transformer_encoder_25/layer_normalization/batchnorm/mul_1Multransformer_encoder_25/add:z:0<transformer_encoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_encoder_25/layer_normalization/batchnorm/mul_2Mul@transformer_encoder_25/layer_normalization/moments/mean:output:0<transformer_encoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Ctransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_encoder_25_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_encoder_25/layer_normalization/batchnorm/subSubKtransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_encoder_25/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_encoder_25/layer_normalization/batchnorm/add_1AddV2>transformer_encoder_25/layer_normalization/batchnorm/mul_1:z:0<transformer_encoder_25/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
5transformer_encoder_25/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_encoder_25_dense_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0u
+transformer_encoder_25/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_encoder_25/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_encoder_25/dense/Tensordot/ShapeShape>transformer_encoder_25/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_encoder_25/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_25/dense/Tensordot/GatherV2GatherV25transformer_encoder_25/dense/Tensordot/Shape:output:04transformer_encoder_25/dense/Tensordot/free:output:0=transformer_encoder_25/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_encoder_25/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_25/dense/Tensordot/GatherV2_1GatherV25transformer_encoder_25/dense/Tensordot/Shape:output:04transformer_encoder_25/dense/Tensordot/axes:output:0?transformer_encoder_25/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_encoder_25/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_encoder_25/dense/Tensordot/ProdProd8transformer_encoder_25/dense/Tensordot/GatherV2:output:05transformer_encoder_25/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_encoder_25/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_25/dense/Tensordot/Prod_1Prod:transformer_encoder_25/dense/Tensordot/GatherV2_1:output:07transformer_encoder_25/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_encoder_25/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_encoder_25/dense/Tensordot/concatConcatV24transformer_encoder_25/dense/Tensordot/free:output:04transformer_encoder_25/dense/Tensordot/axes:output:0;transformer_encoder_25/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_encoder_25/dense/Tensordot/stackPack4transformer_encoder_25/dense/Tensordot/Prod:output:06transformer_encoder_25/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_encoder_25/dense/Tensordot/transpose	Transpose>transformer_encoder_25/layer_normalization/batchnorm/add_1:z:06transformer_encoder_25/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
.transformer_encoder_25/dense/Tensordot/ReshapeReshape4transformer_encoder_25/dense/Tensordot/transpose:y:05transformer_encoder_25/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_encoder_25/dense/Tensordot/MatMulMatMul7transformer_encoder_25/dense/Tensordot/Reshape:output:0=transformer_encoder_25/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
.transformer_encoder_25/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@v
4transformer_encoder_25/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_25/dense/Tensordot/concat_1ConcatV28transformer_encoder_25/dense/Tensordot/GatherV2:output:07transformer_encoder_25/dense/Tensordot/Const_2:output:0=transformer_encoder_25/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_encoder_25/dense/TensordotReshape7transformer_encoder_25/dense/Tensordot/MatMul:product:08transformer_encoder_25/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? @?
3transformer_encoder_25/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_25_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
$transformer_encoder_25/dense/BiasAddBiasAdd/transformer_encoder_25/dense/Tensordot:output:0;transformer_encoder_25/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @?
!transformer_encoder_25/dense/ReluRelu-transformer_encoder_25/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
7transformer_encoder_25/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_encoder_25_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0w
-transformer_encoder_25/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_encoder_25/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_encoder_25/dense_1/Tensordot/ShapeShape/transformer_encoder_25/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_encoder_25/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_25/dense_1/Tensordot/GatherV2GatherV27transformer_encoder_25/dense_1/Tensordot/Shape:output:06transformer_encoder_25/dense_1/Tensordot/free:output:0?transformer_encoder_25/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_encoder_25/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_encoder_25/dense_1/Tensordot/GatherV2_1GatherV27transformer_encoder_25/dense_1/Tensordot/Shape:output:06transformer_encoder_25/dense_1/Tensordot/axes:output:0Atransformer_encoder_25/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_encoder_25/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_25/dense_1/Tensordot/ProdProd:transformer_encoder_25/dense_1/Tensordot/GatherV2:output:07transformer_encoder_25/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_encoder_25/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_encoder_25/dense_1/Tensordot/Prod_1Prod<transformer_encoder_25/dense_1/Tensordot/GatherV2_1:output:09transformer_encoder_25/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_encoder_25/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_25/dense_1/Tensordot/concatConcatV26transformer_encoder_25/dense_1/Tensordot/free:output:06transformer_encoder_25/dense_1/Tensordot/axes:output:0=transformer_encoder_25/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_encoder_25/dense_1/Tensordot/stackPack6transformer_encoder_25/dense_1/Tensordot/Prod:output:08transformer_encoder_25/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_encoder_25/dense_1/Tensordot/transpose	Transpose/transformer_encoder_25/dense/Relu:activations:08transformer_encoder_25/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? @?
0transformer_encoder_25/dense_1/Tensordot/ReshapeReshape6transformer_encoder_25/dense_1/Tensordot/transpose:y:07transformer_encoder_25/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_encoder_25/dense_1/Tensordot/MatMulMatMul9transformer_encoder_25/dense_1/Tensordot/Reshape:output:0?transformer_encoder_25/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? z
0transformer_encoder_25/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: x
6transformer_encoder_25/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_25/dense_1/Tensordot/concat_1ConcatV2:transformer_encoder_25/dense_1/Tensordot/GatherV2:output:09transformer_encoder_25/dense_1/Tensordot/Const_2:output:0?transformer_encoder_25/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_encoder_25/dense_1/TensordotReshape9transformer_encoder_25/dense_1/Tensordot/MatMul:product:0:transformer_encoder_25/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_encoder_25_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&transformer_encoder_25/dense_1/BiasAddBiasAdd1transformer_encoder_25/dense_1/Tensordot:output:0=transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
)transformer_encoder_25/dropout_1/IdentityIdentity/transformer_encoder_25/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
transformer_encoder_25/add_1AddV2>transformer_encoder_25/layer_normalization/batchnorm/add_1:z:02transformer_encoder_25/dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Ktransformer_encoder_25/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_encoder_25/layer_normalization_1/moments/meanMean transformer_encoder_25/add_1:z:0Ttransformer_encoder_25/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_encoder_25/layer_normalization_1/moments/StopGradientStopGradientBtransformer_encoder_25/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_encoder_25/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_encoder_25/add_1:z:0Jtransformer_encoder_25/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Otransformer_encoder_25/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_encoder_25/layer_normalization_1/moments/varianceMeanJtransformer_encoder_25/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_encoder_25/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_encoder_25/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_encoder_25/layer_normalization_1/batchnorm/addAddV2Ftransformer_encoder_25/layer_normalization_1/moments/variance:output:0Etransformer_encoder_25/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_25/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_encoder_25/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_encoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_encoder_25/layer_normalization_1/batchnorm/mulMul@transformer_encoder_25/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
<transformer_encoder_25/layer_normalization_1/batchnorm/mul_1Mul transformer_encoder_25/add_1:z:0>transformer_encoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_encoder_25/layer_normalization_1/batchnorm/mul_2MulBtransformer_encoder_25/layer_normalization_1/moments/mean:output:0>transformer_encoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Etransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_encoder_25_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_encoder_25/layer_normalization_1/batchnorm/subSubMtransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_encoder_25/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_encoder_25/layer_normalization_1/batchnorm/add_1AddV2@transformer_encoder_25/layer_normalization_1/batchnorm/mul_1:z:0>transformer_encoder_25/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
transformer_decoder_25/ShapeShape@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:t
*transformer_decoder_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,transformer_decoder_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,transformer_decoder_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$transformer_decoder_25/strided_sliceStridedSlice%transformer_decoder_25/Shape:output:03transformer_decoder_25/strided_slice/stack:output:05transformer_decoder_25/strided_slice/stack_1:output:05transformer_decoder_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_25/strided_slice_1StridedSlice%transformer_decoder_25/Shape:output:05transformer_decoder_25/strided_slice_1/stack:output:07transformer_decoder_25/strided_slice_1/stack_1:output:07transformer_decoder_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"transformer_decoder_25/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"transformer_decoder_25/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_25/rangeRange+transformer_decoder_25/range/start:output:0/transformer_decoder_25/strided_slice_1:output:0+transformer_decoder_25/range/delta:output:0*
_output_shapes
: }
,transformer_decoder_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&transformer_decoder_25/strided_slice_2StridedSlice%transformer_decoder_25/range:output:05transformer_decoder_25/strided_slice_2/stack:output:07transformer_decoder_25/strided_slice_2/stack_1:output:07transformer_decoder_25/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskf
$transformer_decoder_25/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : f
$transformer_decoder_25/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_25/range_1Range-transformer_decoder_25/range_1/start:output:0/transformer_decoder_25/strided_slice_1:output:0-transformer_decoder_25/range_1/delta:output:0*
_output_shapes
: ?
#transformer_decoder_25/GreaterEqualGreaterEqual/transformer_decoder_25/strided_slice_2:output:0'transformer_decoder_25/range_1:output:0*
T0*
_output_shapes

:  ?
transformer_decoder_25/CastCast'transformer_decoder_25/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  v
,transformer_decoder_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_25/strided_slice_3StridedSlice%transformer_decoder_25/Shape:output:05transformer_decoder_25/strided_slice_3/stack:output:07transformer_decoder_25/strided_slice_3/stack_1:output:07transformer_decoder_25/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_25/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_25/strided_slice_4StridedSlice%transformer_decoder_25/Shape:output:05transformer_decoder_25/strided_slice_4/stack:output:07transformer_decoder_25/strided_slice_4/stack_1:output:07transformer_decoder_25/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&transformer_decoder_25/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
$transformer_decoder_25/Reshape/shapePack/transformer_decoder_25/Reshape/shape/0:output:0/transformer_decoder_25/strided_slice_3:output:0/transformer_decoder_25/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_25/ReshapeReshapetransformer_decoder_25/Cast:y:0-transformer_decoder_25/Reshape/shape:output:0*
T0*"
_output_shapes
:  p
%transformer_decoder_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!transformer_decoder_25/ExpandDims
ExpandDims-transformer_decoder_25/strided_slice:output:0.transformer_decoder_25/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
transformer_decoder_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"      d
"transformer_decoder_25/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
transformer_decoder_25/concatConcatV2*transformer_decoder_25/ExpandDims:output:0%transformer_decoder_25/Const:output:0+transformer_decoder_25/concat/axis:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_25/TileTile'transformer_decoder_25/Reshape:output:0&transformer_decoder_25/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Ntransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_decoder_25/multi_head_attention/query/einsum/EinsumEinsum@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_decoder_25_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_25/multi_head_attention/query/addAddV2Htransformer_decoder_25/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_decoder_25/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_decoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
=transformer_decoder_25/multi_head_attention/key/einsum/EinsumEinsum@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0Ttransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_decoder_25/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_decoder_25_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_decoder_25/multi_head_attention/key/addAddV2Ftransformer_decoder_25/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_decoder_25/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_decoder_25/multi_head_attention/value/einsum/EinsumEinsum@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_decoder_25_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_25/multi_head_attention/value/addAddV2Htransformer_decoder_25/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_decoder_25/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_decoder_25/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
/transformer_decoder_25/multi_head_attention/MulMul9transformer_decoder_25/multi_head_attention/query/add:z:0:transformer_decoder_25/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_decoder_25/multi_head_attention/einsum/EinsumEinsum7transformer_decoder_25/multi_head_attention/key/add:z:03transformer_decoder_25/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
:transformer_decoder_25/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6transformer_decoder_25/multi_head_attention/ExpandDims
ExpandDims$transformer_decoder_25/Tile:output:0Ctransformer_decoder_25/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
8transformer_decoder_25/multi_head_attention/softmax/CastCast?transformer_decoder_25/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ~
9transformer_decoder_25/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7transformer_decoder_25/multi_head_attention/softmax/subSubBtransformer_decoder_25/multi_head_attention/softmax/sub/x:output:0<transformer_decoder_25/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ~
9transformer_decoder_25/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
7transformer_decoder_25/multi_head_attention/softmax/mulMul;transformer_decoder_25/multi_head_attention/softmax/sub:z:0Btransformer_decoder_25/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
7transformer_decoder_25/multi_head_attention/softmax/addAddV2Btransformer_decoder_25/multi_head_attention/einsum/Einsum:output:0;transformer_decoder_25/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_25/multi_head_attention/softmax/SoftmaxSoftmax;transformer_decoder_25/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
>transformer_decoder_25/multi_head_attention/dropout_2/IdentityIdentityEtransformer_decoder_25/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_25/multi_head_attention/einsum_1/EinsumEinsumGtransformer_decoder_25/multi_head_attention/dropout_2/Identity:output:09transformer_decoder_25/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_decoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Jtransformer_decoder_25/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_decoder_25/multi_head_attention/einsum_1/Einsum:output:0atransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
Otransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_decoder_25_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
@transformer_decoder_25/multi_head_attention/attention_output/addAddV2Stransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
'transformer_decoder_25/dropout/IdentityIdentityDtransformer_decoder_25/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  ?
transformer_decoder_25/addAddV20transformer_decoder_25/dropout/Identity:output:0@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????  ?
Itransformer_decoder_25/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_decoder_25/layer_normalization/moments/meanMeantransformer_decoder_25/add:z:0Rtransformer_decoder_25/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_decoder_25/layer_normalization/moments/StopGradientStopGradient@transformer_decoder_25/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_decoder_25/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_25/add:z:0Htransformer_decoder_25/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Mtransformer_decoder_25/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_decoder_25/layer_normalization/moments/varianceMeanHtransformer_decoder_25/layer_normalization/moments/SquaredDifference:z:0Vtransformer_decoder_25/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_decoder_25/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_decoder_25/layer_normalization/batchnorm/addAddV2Dtransformer_decoder_25/layer_normalization/moments/variance:output:0Ctransformer_decoder_25/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_25/layer_normalization/batchnorm/RsqrtRsqrt<transformer_decoder_25/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_decoder_25_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_decoder_25/layer_normalization/batchnorm/mulMul>transformer_decoder_25/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
:transformer_decoder_25/layer_normalization/batchnorm/mul_1Multransformer_decoder_25/add:z:0<transformer_decoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_decoder_25/layer_normalization/batchnorm/mul_2Mul@transformer_decoder_25/layer_normalization/moments/mean:output:0<transformer_decoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Ctransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_decoder_25_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_decoder_25/layer_normalization/batchnorm/subSubKtransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_decoder_25/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_decoder_25/layer_normalization/batchnorm/add_1AddV2>transformer_decoder_25/layer_normalization/batchnorm/mul_1:z:0<transformer_decoder_25/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
5transformer_decoder_25/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_decoder_25_dense_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0u
+transformer_decoder_25/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_decoder_25/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_decoder_25/dense/Tensordot/ShapeShape>transformer_decoder_25/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_decoder_25/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_25/dense/Tensordot/GatherV2GatherV25transformer_decoder_25/dense/Tensordot/Shape:output:04transformer_decoder_25/dense/Tensordot/free:output:0=transformer_decoder_25/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_decoder_25/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_25/dense/Tensordot/GatherV2_1GatherV25transformer_decoder_25/dense/Tensordot/Shape:output:04transformer_decoder_25/dense/Tensordot/axes:output:0?transformer_decoder_25/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_decoder_25/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_decoder_25/dense/Tensordot/ProdProd8transformer_decoder_25/dense/Tensordot/GatherV2:output:05transformer_decoder_25/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_decoder_25/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_25/dense/Tensordot/Prod_1Prod:transformer_decoder_25/dense/Tensordot/GatherV2_1:output:07transformer_decoder_25/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_decoder_25/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_decoder_25/dense/Tensordot/concatConcatV24transformer_decoder_25/dense/Tensordot/free:output:04transformer_decoder_25/dense/Tensordot/axes:output:0;transformer_decoder_25/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_decoder_25/dense/Tensordot/stackPack4transformer_decoder_25/dense/Tensordot/Prod:output:06transformer_decoder_25/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_decoder_25/dense/Tensordot/transpose	Transpose>transformer_decoder_25/layer_normalization/batchnorm/add_1:z:06transformer_decoder_25/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
.transformer_decoder_25/dense/Tensordot/ReshapeReshape4transformer_decoder_25/dense/Tensordot/transpose:y:05transformer_decoder_25/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_decoder_25/dense/Tensordot/MatMulMatMul7transformer_decoder_25/dense/Tensordot/Reshape:output:0=transformer_decoder_25/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
.transformer_decoder_25/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@v
4transformer_decoder_25/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_25/dense/Tensordot/concat_1ConcatV28transformer_decoder_25/dense/Tensordot/GatherV2:output:07transformer_decoder_25/dense/Tensordot/Const_2:output:0=transformer_decoder_25/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_decoder_25/dense/TensordotReshape7transformer_decoder_25/dense/Tensordot/MatMul:product:08transformer_decoder_25/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? @?
3transformer_decoder_25/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_decoder_25_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
$transformer_decoder_25/dense/BiasAddBiasAdd/transformer_decoder_25/dense/Tensordot:output:0;transformer_decoder_25/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @?
!transformer_decoder_25/dense/ReluRelu-transformer_decoder_25/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
7transformer_decoder_25/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_decoder_25_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0w
-transformer_decoder_25/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_decoder_25/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_decoder_25/dense_1/Tensordot/ShapeShape/transformer_decoder_25/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_decoder_25/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_25/dense_1/Tensordot/GatherV2GatherV27transformer_decoder_25/dense_1/Tensordot/Shape:output:06transformer_decoder_25/dense_1/Tensordot/free:output:0?transformer_decoder_25/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_decoder_25/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_decoder_25/dense_1/Tensordot/GatherV2_1GatherV27transformer_decoder_25/dense_1/Tensordot/Shape:output:06transformer_decoder_25/dense_1/Tensordot/axes:output:0Atransformer_decoder_25/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_decoder_25/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_25/dense_1/Tensordot/ProdProd:transformer_decoder_25/dense_1/Tensordot/GatherV2:output:07transformer_decoder_25/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_decoder_25/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_decoder_25/dense_1/Tensordot/Prod_1Prod<transformer_decoder_25/dense_1/Tensordot/GatherV2_1:output:09transformer_decoder_25/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_decoder_25/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_25/dense_1/Tensordot/concatConcatV26transformer_decoder_25/dense_1/Tensordot/free:output:06transformer_decoder_25/dense_1/Tensordot/axes:output:0=transformer_decoder_25/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_decoder_25/dense_1/Tensordot/stackPack6transformer_decoder_25/dense_1/Tensordot/Prod:output:08transformer_decoder_25/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_decoder_25/dense_1/Tensordot/transpose	Transpose/transformer_decoder_25/dense/Relu:activations:08transformer_decoder_25/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? @?
0transformer_decoder_25/dense_1/Tensordot/ReshapeReshape6transformer_decoder_25/dense_1/Tensordot/transpose:y:07transformer_decoder_25/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_decoder_25/dense_1/Tensordot/MatMulMatMul9transformer_decoder_25/dense_1/Tensordot/Reshape:output:0?transformer_decoder_25/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? z
0transformer_decoder_25/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: x
6transformer_decoder_25/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_25/dense_1/Tensordot/concat_1ConcatV2:transformer_decoder_25/dense_1/Tensordot/GatherV2:output:09transformer_decoder_25/dense_1/Tensordot/Const_2:output:0?transformer_decoder_25/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_decoder_25/dense_1/TensordotReshape9transformer_decoder_25/dense_1/Tensordot/MatMul:product:0:transformer_decoder_25/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_decoder_25_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&transformer_decoder_25/dense_1/BiasAddBiasAdd1transformer_decoder_25/dense_1/Tensordot:output:0=transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
)transformer_decoder_25/dropout_1/IdentityIdentity/transformer_decoder_25/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
transformer_decoder_25/add_1AddV2>transformer_decoder_25/layer_normalization/batchnorm/add_1:z:02transformer_decoder_25/dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Ktransformer_decoder_25/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_decoder_25/layer_normalization_1/moments/meanMean transformer_decoder_25/add_1:z:0Ttransformer_decoder_25/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_decoder_25/layer_normalization_1/moments/StopGradientStopGradientBtransformer_decoder_25/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_decoder_25/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_decoder_25/add_1:z:0Jtransformer_decoder_25/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Otransformer_decoder_25/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_decoder_25/layer_normalization_1/moments/varianceMeanJtransformer_decoder_25/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_decoder_25/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_decoder_25/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_decoder_25/layer_normalization_1/batchnorm/addAddV2Ftransformer_decoder_25/layer_normalization_1/moments/variance:output:0Etransformer_decoder_25/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_25/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_decoder_25/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_decoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_decoder_25/layer_normalization_1/batchnorm/mulMul@transformer_decoder_25/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
<transformer_decoder_25/layer_normalization_1/batchnorm/mul_1Mul transformer_decoder_25/add_1:z:0>transformer_decoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_decoder_25/layer_normalization_1/batchnorm/mul_2MulBtransformer_decoder_25/layer_normalization_1/moments/mean:output:0>transformer_decoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Etransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_decoder_25_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_decoder_25/layer_normalization_1/batchnorm/subSubMtransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_decoder_25/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_decoder_25/layer_normalization_1/batchnorm/add_1AddV2@transformer_decoder_25/layer_normalization_1/batchnorm/mul_1:z:0>transformer_decoder_25/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  s
1global_average_pooling1d_9/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_9/MeanMean@transformer_decoder_25/layer_normalization_1/batchnorm/add_1:z:0:global_average_pooling1d_9/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
dense_14/MatMulMatMul(global_average_pooling1d_9/Mean:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? h
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:????????? i
IdentityIdentitydense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2D^token_and_position_embedding_23/position_embedding24/ReadVariableOpC^token_and_position_embedding_23/token_embedding24/embedding_lookupD^token_and_position_embedding_24/position_embedding25/ReadVariableOpC^token_and_position_embedding_24/token_embedding25/embedding_lookup4^transformer_decoder_25/dense/BiasAdd/ReadVariableOp6^transformer_decoder_25/dense/Tensordot/ReadVariableOp6^transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp8^transformer_decoder_25/dense_1/Tensordot/ReadVariableOpD^transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpH^transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_decoder_25/multi_head_attention/key/add/ReadVariableOpM^transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_decoder_25/multi_head_attention/query/add/ReadVariableOpO^transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_decoder_25/multi_head_attention/value/add/ReadVariableOpO^transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp4^transformer_encoder_25/dense/BiasAdd/ReadVariableOp6^transformer_encoder_25/dense/Tensordot/ReadVariableOp6^transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp8^transformer_encoder_25/dense_1/Tensordot/ReadVariableOpD^transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpH^transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_encoder_25/multi_head_attention/key/add/ReadVariableOpM^transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_encoder_25/multi_head_attention/query/add/ReadVariableOpO^transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_encoder_25/multi_head_attention/value/add/ReadVariableOpO^transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2?
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22?
Ctoken_and_position_embedding_23/position_embedding24/ReadVariableOpCtoken_and_position_embedding_23/position_embedding24/ReadVariableOp2?
Btoken_and_position_embedding_23/token_embedding24/embedding_lookupBtoken_and_position_embedding_23/token_embedding24/embedding_lookup2?
Ctoken_and_position_embedding_24/position_embedding25/ReadVariableOpCtoken_and_position_embedding_24/position_embedding25/ReadVariableOp2?
Btoken_and_position_embedding_24/token_embedding25/embedding_lookupBtoken_and_position_embedding_24/token_embedding25/embedding_lookup2j
3transformer_decoder_25/dense/BiasAdd/ReadVariableOp3transformer_decoder_25/dense/BiasAdd/ReadVariableOp2n
5transformer_decoder_25/dense/Tensordot/ReadVariableOp5transformer_decoder_25/dense/Tensordot/ReadVariableOp2n
5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp2r
7transformer_decoder_25/dense_1/Tensordot/ReadVariableOp7transformer_decoder_25/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpCtransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_decoder_25/multi_head_attention/key/add/ReadVariableOpBtransformer_decoder_25/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOpDtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOpDtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp2j
3transformer_encoder_25/dense/BiasAdd/ReadVariableOp3transformer_encoder_25/dense/BiasAdd/ReadVariableOp2n
5transformer_encoder_25/dense/Tensordot/ReadVariableOp5transformer_encoder_25/dense/Tensordot/ReadVariableOp2n
5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp2r
7transformer_encoder_25/dense_1/Tensordot/ReadVariableOp7transformer_encoder_25/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpCtransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_encoder_25/multi_head_attention/key/add/ReadVariableOpBtransformer_encoder_25/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOpDtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOpDtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:Q M
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
?
?
@__inference_token_and_position_embedding_23_layer_call_fn_503670

inputs	
unknown:	? 
	unknown_0:  
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_500760s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????  `
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
?
G
__inference__creator_504590
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_340481*
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
?
?	
(__inference_model_9_layer_call_fn_502217

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	? 
	unknown_4:  
	unknown_5: 
	unknown_6:  
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22:  

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39:  

unknown_40: 
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
unknown_40*7
Tin0
.2,		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *H
_read_only_resource_inputs*
(&	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_9_layer_call_and_return_conditional_losses_502040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
ܒ
?
C__inference_model_9_layer_call_and_return_conditional_losses_502040

inputs
inputs_1S
Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_9_string_lookup_9_equal_y3
/text_vectorization_9_string_lookup_9_selectv2_t	9
&token_and_position_embedding_23_501956:	? 8
&token_and_position_embedding_23_501958:  8
&token_and_position_embedding_24_501961: 8
&token_and_position_embedding_24_501963:  3
transformer_encoder_25_501967: /
transformer_encoder_25_501969:3
transformer_encoder_25_501971: /
transformer_encoder_25_501973:3
transformer_encoder_25_501975: /
transformer_encoder_25_501977:3
transformer_encoder_25_501979: +
transformer_encoder_25_501981: +
transformer_encoder_25_501983: +
transformer_encoder_25_501985: /
transformer_encoder_25_501987: @+
transformer_encoder_25_501989:@/
transformer_encoder_25_501991:@ +
transformer_encoder_25_501993: +
transformer_encoder_25_501995: +
transformer_encoder_25_501997: 3
transformer_decoder_25_502000: /
transformer_decoder_25_502002:3
transformer_decoder_25_502004: /
transformer_decoder_25_502006:3
transformer_decoder_25_502008: /
transformer_decoder_25_502010:3
transformer_decoder_25_502012: +
transformer_decoder_25_502014: +
transformer_decoder_25_502016: +
transformer_decoder_25_502018: /
transformer_decoder_25_502020: @+
transformer_decoder_25_502022:@/
transformer_decoder_25_502024:@ +
transformer_decoder_25_502026: +
transformer_decoder_25_502028: +
transformer_decoder_25_502030: !
dense_14_502034:  
dense_14_502036: 
identity?? dense_14/StatefulPartitionedCall?Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?7token_and_position_embedding_23/StatefulPartitionedCall?7token_and_position_embedding_24/StatefulPartitionedCall?.transformer_decoder_25/StatefulPartitionedCall?.transformer_encoder_25/StatefulPartitionedCall}
text_vectorization_9/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_9/StringSplit/StringSplitV2StringSplitV2%text_vectorization_9/Squeeze:output:0/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_9/StringSplit/strided_sliceStridedSlice8text_vectorization_9/StringSplit/StringSplitV2:indices:0=text_vectorization_9/StringSplit/strided_slice/stack:output:0?text_vectorization_9/StringSplit/strided_slice/stack_1:output:0?text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_9/StringSplit/strided_slice_1StridedSlice6text_vectorization_9/StringSplit/StringSplitV2:shape:0?text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
itext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
dtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_9/StringSplit/StringSplitV2:values:0Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_9/string_lookup_9/EqualEqual7text_vectorization_9/StringSplit/StringSplitV2:values:0,text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/SelectV2SelectV2.text_vectorization_9/string_lookup_9/Equal:z:0/text_vectorization_9_string_lookup_9_selectv2_tKtext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/IdentityIdentity6text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
8text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_9/RaggedToTensor/Const:output:06text_vectorization_9/string_lookup_9/Identity:output:0:text_vectorization_9/RaggedToTensor/default_value:output:09text_vectorization_9/StringSplit/strided_slice_1:output:07text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_23/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_23_501956&token_and_position_embedding_23_501958*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_500760?
7token_and_position_embedding_24/StatefulPartitionedCallStatefulPartitionedCallinputs_1&token_and_position_embedding_24_501961&token_and_position_embedding_24_501963*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_500794?
add_9/PartitionedCallPartitionedCall@token_and_position_embedding_23/StatefulPartitionedCall:output:0@token_and_position_embedding_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_500806?
.transformer_encoder_25/StatefulPartitionedCallStatefulPartitionedCalladd_9/PartitionedCall:output:0transformer_encoder_25_501967transformer_encoder_25_501969transformer_encoder_25_501971transformer_encoder_25_501973transformer_encoder_25_501975transformer_encoder_25_501977transformer_encoder_25_501979transformer_encoder_25_501981transformer_encoder_25_501983transformer_encoder_25_501985transformer_encoder_25_501987transformer_encoder_25_501989transformer_encoder_25_501991transformer_encoder_25_501993transformer_encoder_25_501995transformer_encoder_25_501997*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_501749?
.transformer_decoder_25/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_25/StatefulPartitionedCall:output:0transformer_decoder_25_502000transformer_decoder_25_502002transformer_decoder_25_502004transformer_decoder_25_502006transformer_decoder_25_502008transformer_decoder_25_502010transformer_decoder_25_502012transformer_decoder_25_502014transformer_decoder_25_502016transformer_decoder_25_502018transformer_decoder_25_502020transformer_decoder_25_502022transformer_decoder_25_502024transformer_decoder_25_502026transformer_decoder_25_502028transformer_decoder_25_502030*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501526?
*global_average_pooling1d_9/PartitionedCallPartitionedCall7transformer_decoder_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_500673?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_9/PartitionedCall:output:0dense_14_502034dense_14_502036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_501188x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_14/StatefulPartitionedCallC^text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV28^token_and_position_embedding_23/StatefulPartitionedCall8^token_and_position_embedding_24/StatefulPartitionedCall/^transformer_decoder_25/StatefulPartitionedCall/^transformer_encoder_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_23/StatefulPartitionedCall7token_and_position_embedding_23/StatefulPartitionedCall2r
7token_and_position_embedding_24/StatefulPartitionedCall7token_and_position_embedding_24/StatefulPartitionedCall2`
.transformer_decoder_25/StatefulPartitionedCall.transformer_decoder_25/StatefulPartitionedCall2`
.transformer_encoder_25/StatefulPartitionedCall.transformer_encoder_25/StatefulPartitionedCall:O K
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
?
?
7__inference_transformer_decoder_25_layer_call_fn_504169
decoder_sequence
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: @

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 
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
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501526s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????  
*
_user_specified_namedecoder_sequence
?"
?
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_503734

inputs;
)token_embedding25_embedding_lookup_503710: >
,position_embedding25_readvariableop_resource:  
identity??#position_embedding25/ReadVariableOp?"token_embedding25/embedding_lookupg
token_embedding25/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
"token_embedding25/embedding_lookupResourceGather)token_embedding25_embedding_lookup_503710token_embedding25/Cast:y:0*
Tindices0*<
_class2
0.loc:@token_embedding25/embedding_lookup/503710*+
_output_shapes
:?????????  *
dtype0?
+token_embedding25/embedding_lookup/IdentityIdentity+token_embedding25/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding25/embedding_lookup/503710*+
_output_shapes
:?????????  ?
-token_embedding25/embedding_lookup/Identity_1Identity4token_embedding25/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
position_embedding25/ShapeShape6token_embedding25/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding25/strided_sliceStridedSlice#position_embedding25/Shape:output:01position_embedding25/strided_slice/stack:output:03position_embedding25/strided_slice/stack_1:output:03position_embedding25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding25/ReadVariableOpReadVariableOp,position_embedding25_readvariableop_resource*
_output_shapes

:  *
dtype0\
position_embedding25/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding25/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding25/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding25/strided_slice_1/stackPack#position_embedding25/Const:output:05position_embedding25/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding25/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding25/strided_slice_1/stack_1Pack+position_embedding25/strided_slice:output:07position_embedding25/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding25/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding25/strided_slice_1/stack_2Pack%position_embedding25/Const_1:output:07position_embedding25/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding25/strided_slice_1StridedSlice+position_embedding25/ReadVariableOp:value:03position_embedding25/strided_slice_1/stack:output:05position_embedding25/strided_slice_1/stack_1:output:05position_embedding25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
 position_embedding25/BroadcastToBroadcastTo-position_embedding25/strided_slice_1:output:0#position_embedding25/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
addAddV26token_embedding25/embedding_lookup/Identity_1:output:0)position_embedding25/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp$^position_embedding25/ReadVariableOp#^token_embedding25/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding25/ReadVariableOp#position_embedding25/ReadVariableOp2H
"token_embedding25/embedding_lookup"token_embedding25/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?4
C__inference_model_9_layer_call_and_return_conditional_losses_503661
inputs_0
inputs_1S
Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_9_string_lookup_9_equal_y3
/text_vectorization_9_string_lookup_9_selectv2_t	\
Itoken_and_position_embedding_23_token_embedding24_embedding_lookup_503269:	? ^
Ltoken_and_position_embedding_23_position_embedding24_readvariableop_resource:  [
Itoken_and_position_embedding_24_token_embedding25_embedding_lookup_503293: ^
Ltoken_and_position_embedding_24_position_embedding25_readvariableop_resource:  m
Wtransformer_encoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource: _
Mtransformer_encoder_25_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_encoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource: ]
Ktransformer_encoder_25_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_encoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource: _
Mtransformer_encoder_25_multi_head_attention_value_add_readvariableop_resource:x
btransformer_encoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: f
Xtransformer_encoder_25_multi_head_attention_attention_output_add_readvariableop_resource: ^
Ptransformer_encoder_25_layer_normalization_batchnorm_mul_readvariableop_resource: Z
Ltransformer_encoder_25_layer_normalization_batchnorm_readvariableop_resource: P
>transformer_encoder_25_dense_tensordot_readvariableop_resource: @J
<transformer_encoder_25_dense_biasadd_readvariableop_resource:@R
@transformer_encoder_25_dense_1_tensordot_readvariableop_resource:@ L
>transformer_encoder_25_dense_1_biasadd_readvariableop_resource: `
Rtransformer_encoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource: \
Ntransformer_encoder_25_layer_normalization_1_batchnorm_readvariableop_resource: m
Wtransformer_decoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource: _
Mtransformer_decoder_25_multi_head_attention_query_add_readvariableop_resource:k
Utransformer_decoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource: ]
Ktransformer_decoder_25_multi_head_attention_key_add_readvariableop_resource:m
Wtransformer_decoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource: _
Mtransformer_decoder_25_multi_head_attention_value_add_readvariableop_resource:x
btransformer_decoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: f
Xtransformer_decoder_25_multi_head_attention_attention_output_add_readvariableop_resource: ^
Ptransformer_decoder_25_layer_normalization_batchnorm_mul_readvariableop_resource: Z
Ltransformer_decoder_25_layer_normalization_batchnorm_readvariableop_resource: P
>transformer_decoder_25_dense_tensordot_readvariableop_resource: @J
<transformer_decoder_25_dense_biasadd_readvariableop_resource:@R
@transformer_decoder_25_dense_1_tensordot_readvariableop_resource:@ L
>transformer_decoder_25_dense_1_biasadd_readvariableop_resource: `
Rtransformer_decoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource: \
Ntransformer_decoder_25_layer_normalization_1_batchnorm_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource:  6
(dense_14_biasadd_readvariableop_resource: 
identity??dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?Ctoken_and_position_embedding_23/position_embedding24/ReadVariableOp?Btoken_and_position_embedding_23/token_embedding24/embedding_lookup?Ctoken_and_position_embedding_24/position_embedding25/ReadVariableOp?Btoken_and_position_embedding_24/token_embedding25/embedding_lookup?3transformer_decoder_25/dense/BiasAdd/ReadVariableOp?5transformer_decoder_25/dense/Tensordot/ReadVariableOp?5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp?7transformer_decoder_25/dense_1/Tensordot/ReadVariableOp?Ctransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_decoder_25/multi_head_attention/key/add/ReadVariableOp?Ltransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOp?Ntransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOp?Ntransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp?3transformer_encoder_25/dense/BiasAdd/ReadVariableOp?5transformer_encoder_25/dense/Tensordot/ReadVariableOp?5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp?7transformer_encoder_25/dense_1/Tensordot/ReadVariableOp?Ctransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp?Gtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp?Etransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp?Itransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp?Otransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp?Ytransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Btransformer_encoder_25/multi_head_attention/key/add/ReadVariableOp?Ltransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOp?Ntransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Dtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOp?Ntransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp
text_vectorization_9/SqueezeSqueezeinputs_0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_9/StringSplit/StringSplitV2StringSplitV2%text_vectorization_9/Squeeze:output:0/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_9/StringSplit/strided_sliceStridedSlice8text_vectorization_9/StringSplit/StringSplitV2:indices:0=text_vectorization_9/StringSplit/strided_slice/stack:output:0?text_vectorization_9/StringSplit/strided_slice/stack_1:output:0?text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_9/StringSplit/strided_slice_1StridedSlice6text_vectorization_9/StringSplit/StringSplitV2:shape:0?text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
itext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
dtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_9/StringSplit/StringSplitV2:values:0Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_9/string_lookup_9/EqualEqual7text_vectorization_9/StringSplit/StringSplitV2:values:0,text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/SelectV2SelectV2.text_vectorization_9/string_lookup_9/Equal:z:0/text_vectorization_9_string_lookup_9_selectv2_tKtext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/IdentityIdentity6text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
8text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_9/RaggedToTensor/Const:output:06text_vectorization_9/string_lookup_9/Identity:output:0:text_vectorization_9/RaggedToTensor/default_value:output:09text_vectorization_9/StringSplit/strided_slice_1:output:07text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Btoken_and_position_embedding_23/token_embedding24/embedding_lookupResourceGatherItoken_and_position_embedding_23_token_embedding24_embedding_lookup_503269Atext_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*\
_classR
PNloc:@token_and_position_embedding_23/token_embedding24/embedding_lookup/503269*+
_output_shapes
:?????????  *
dtype0?
Ktoken_and_position_embedding_23/token_embedding24/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_23/token_embedding24/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_23/token_embedding24/embedding_lookup/503269*+
_output_shapes
:?????????  ?
Mtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
:token_and_position_embedding_23/position_embedding24/ShapeShapeVtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_23/position_embedding24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_23/position_embedding24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_23/position_embedding24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_23/position_embedding24/strided_sliceStridedSliceCtoken_and_position_embedding_23/position_embedding24/Shape:output:0Qtoken_and_position_embedding_23/position_embedding24/strided_slice/stack:output:0Stoken_and_position_embedding_23/position_embedding24/strided_slice/stack_1:output:0Stoken_and_position_embedding_23/position_embedding24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_23/position_embedding24/ReadVariableOpReadVariableOpLtoken_and_position_embedding_23_position_embedding24_readvariableop_resource*
_output_shapes

:  *
dtype0|
:token_and_position_embedding_23/position_embedding24/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_23/position_embedding24/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_23/position_embedding24/strided_slice_1/stackPackCtoken_and_position_embedding_23/position_embedding24/Const:output:0Utoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1PackKtoken_and_position_embedding_23/position_embedding24/strided_slice:output:0Wtoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2PackEtoken_and_position_embedding_23/position_embedding24/Const_1:output:0Wtoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_23/position_embedding24/strided_slice_1StridedSliceKtoken_and_position_embedding_23/position_embedding24/ReadVariableOp:value:0Stoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack:output:0Utoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
@token_and_position_embedding_23/position_embedding24/BroadcastToBroadcastToMtoken_and_position_embedding_23/position_embedding24/strided_slice_1:output:0Ctoken_and_position_embedding_23/position_embedding24/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
#token_and_position_embedding_23/addAddV2Vtoken_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_23/position_embedding24/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  ?
6token_and_position_embedding_24/token_embedding25/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
Btoken_and_position_embedding_24/token_embedding25/embedding_lookupResourceGatherItoken_and_position_embedding_24_token_embedding25_embedding_lookup_503293:token_and_position_embedding_24/token_embedding25/Cast:y:0*
Tindices0*\
_classR
PNloc:@token_and_position_embedding_24/token_embedding25/embedding_lookup/503293*+
_output_shapes
:?????????  *
dtype0?
Ktoken_and_position_embedding_24/token_embedding25/embedding_lookup/IdentityIdentityKtoken_and_position_embedding_24/token_embedding25/embedding_lookup:output:0*
T0*\
_classR
PNloc:@token_and_position_embedding_24/token_embedding25/embedding_lookup/503293*+
_output_shapes
:?????????  ?
Mtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1IdentityTtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
:token_and_position_embedding_24/position_embedding25/ShapeShapeVtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Htoken_and_position_embedding_24/position_embedding25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_24/position_embedding25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Jtoken_and_position_embedding_24/position_embedding25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Btoken_and_position_embedding_24/position_embedding25/strided_sliceStridedSliceCtoken_and_position_embedding_24/position_embedding25/Shape:output:0Qtoken_and_position_embedding_24/position_embedding25/strided_slice/stack:output:0Stoken_and_position_embedding_24/position_embedding25/strided_slice/stack_1:output:0Stoken_and_position_embedding_24/position_embedding25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ctoken_and_position_embedding_24/position_embedding25/ReadVariableOpReadVariableOpLtoken_and_position_embedding_24_position_embedding25_readvariableop_resource*
_output_shapes

:  *
dtype0|
:token_and_position_embedding_24/position_embedding25/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<token_and_position_embedding_24/position_embedding25/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_24/position_embedding25/strided_slice_1/stackPackCtoken_and_position_embedding_24/position_embedding25/Const:output:0Utoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Ltoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1PackKtoken_and_position_embedding_24/position_embedding25/strided_slice:output:0Wtoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ntoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Ltoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2PackEtoken_and_position_embedding_24/position_embedding25/Const_1:output:0Wtoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Dtoken_and_position_embedding_24/position_embedding25/strided_slice_1StridedSliceKtoken_and_position_embedding_24/position_embedding25/ReadVariableOp:value:0Stoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack:output:0Utoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1:output:0Utoken_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
@token_and_position_embedding_24/position_embedding25/BroadcastToBroadcastToMtoken_and_position_embedding_24/position_embedding25/strided_slice_1:output:0Ctoken_and_position_embedding_24/position_embedding25/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
#token_and_position_embedding_24/addAddV2Vtoken_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1:output:0Itoken_and_position_embedding_24/position_embedding25/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  ?
	add_9/addAddV2'token_and_position_embedding_23/add:z:0'token_and_position_embedding_24/add:z:0*
T0*+
_output_shapes
:?????????  ?
Ntransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_encoder_25/multi_head_attention/query/einsum/EinsumEinsumadd_9/add:z:0Vtransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_encoder_25_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_25/multi_head_attention/query/addAddV2Htransformer_encoder_25/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_encoder_25/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_encoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
=transformer_encoder_25/multi_head_attention/key/einsum/EinsumEinsumadd_9/add:z:0Ttransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_encoder_25/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_encoder_25_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_encoder_25/multi_head_attention/key/addAddV2Ftransformer_encoder_25/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_encoder_25/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_encoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_encoder_25/multi_head_attention/value/einsum/EinsumEinsumadd_9/add:z:0Vtransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_encoder_25_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_encoder_25/multi_head_attention/value/addAddV2Htransformer_encoder_25/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_encoder_25/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_encoder_25/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
/transformer_encoder_25/multi_head_attention/MulMul9transformer_encoder_25/multi_head_attention/query/add:z:0:transformer_encoder_25/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_encoder_25/multi_head_attention/einsum/EinsumEinsum7transformer_encoder_25/multi_head_attention/key/add:z:03transformer_encoder_25/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
;transformer_encoder_25/multi_head_attention/softmax/SoftmaxSoftmaxBtransformer_encoder_25/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_encoder_25/multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
Atransformer_encoder_25/multi_head_attention/dropout_2/dropout/MulMulEtransformer_encoder_25/multi_head_attention/softmax/Softmax:softmax:0Ltransformer_encoder_25/multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_encoder_25/multi_head_attention/dropout_2/dropout/ShapeShapeEtransformer_encoder_25/multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Ztransformer_encoder_25/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniformLtransformer_encoder_25/multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0?
Ltransformer_encoder_25/multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Jtransformer_encoder_25/multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualctransformer_encoder_25/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0Utransformer_encoder_25/multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_encoder_25/multi_head_attention/dropout_2/dropout/CastCastNtransformer_encoder_25/multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
Ctransformer_encoder_25/multi_head_attention/dropout_2/dropout/Mul_1MulEtransformer_encoder_25/multi_head_attention/dropout_2/dropout/Mul:z:0Ftransformer_encoder_25/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
;transformer_encoder_25/multi_head_attention/einsum_1/EinsumEinsumGtransformer_encoder_25/multi_head_attention/dropout_2/dropout/Mul_1:z:09transformer_encoder_25/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_encoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Jtransformer_encoder_25/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_encoder_25/multi_head_attention/einsum_1/Einsum:output:0atransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
Otransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_encoder_25_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
@transformer_encoder_25/multi_head_attention/attention_output/addAddV2Stransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  q
,transformer_encoder_25/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*transformer_encoder_25/dropout/dropout/MulMulDtransformer_encoder_25/multi_head_attention/attention_output/add:z:05transformer_encoder_25/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  ?
,transformer_encoder_25/dropout/dropout/ShapeShapeDtransformer_encoder_25/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
Ctransformer_encoder_25/dropout/dropout/random_uniform/RandomUniformRandomUniform5transformer_encoder_25/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
dtype0z
5transformer_encoder_25/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3transformer_encoder_25/dropout/dropout/GreaterEqualGreaterEqualLtransformer_encoder_25/dropout/dropout/random_uniform/RandomUniform:output:0>transformer_encoder_25/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????  ?
+transformer_encoder_25/dropout/dropout/CastCast7transformer_encoder_25/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
,transformer_encoder_25/dropout/dropout/Mul_1Mul.transformer_encoder_25/dropout/dropout/Mul:z:0/transformer_encoder_25/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
transformer_encoder_25/addAddV2add_9/add:z:00transformer_encoder_25/dropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ?
Itransformer_encoder_25/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_encoder_25/layer_normalization/moments/meanMeantransformer_encoder_25/add:z:0Rtransformer_encoder_25/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_encoder_25/layer_normalization/moments/StopGradientStopGradient@transformer_encoder_25/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_encoder_25/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_25/add:z:0Htransformer_encoder_25/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Mtransformer_encoder_25/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_encoder_25/layer_normalization/moments/varianceMeanHtransformer_encoder_25/layer_normalization/moments/SquaredDifference:z:0Vtransformer_encoder_25/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_encoder_25/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_encoder_25/layer_normalization/batchnorm/addAddV2Dtransformer_encoder_25/layer_normalization/moments/variance:output:0Ctransformer_encoder_25/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_encoder_25/layer_normalization/batchnorm/RsqrtRsqrt<transformer_encoder_25/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_encoder_25_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_encoder_25/layer_normalization/batchnorm/mulMul>transformer_encoder_25/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
:transformer_encoder_25/layer_normalization/batchnorm/mul_1Multransformer_encoder_25/add:z:0<transformer_encoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_encoder_25/layer_normalization/batchnorm/mul_2Mul@transformer_encoder_25/layer_normalization/moments/mean:output:0<transformer_encoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Ctransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_encoder_25_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_encoder_25/layer_normalization/batchnorm/subSubKtransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_encoder_25/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_encoder_25/layer_normalization/batchnorm/add_1AddV2>transformer_encoder_25/layer_normalization/batchnorm/mul_1:z:0<transformer_encoder_25/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
5transformer_encoder_25/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_encoder_25_dense_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0u
+transformer_encoder_25/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_encoder_25/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_encoder_25/dense/Tensordot/ShapeShape>transformer_encoder_25/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_encoder_25/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_25/dense/Tensordot/GatherV2GatherV25transformer_encoder_25/dense/Tensordot/Shape:output:04transformer_encoder_25/dense/Tensordot/free:output:0=transformer_encoder_25/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_encoder_25/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_25/dense/Tensordot/GatherV2_1GatherV25transformer_encoder_25/dense/Tensordot/Shape:output:04transformer_encoder_25/dense/Tensordot/axes:output:0?transformer_encoder_25/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_encoder_25/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_encoder_25/dense/Tensordot/ProdProd8transformer_encoder_25/dense/Tensordot/GatherV2:output:05transformer_encoder_25/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_encoder_25/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_25/dense/Tensordot/Prod_1Prod:transformer_encoder_25/dense/Tensordot/GatherV2_1:output:07transformer_encoder_25/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_encoder_25/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_encoder_25/dense/Tensordot/concatConcatV24transformer_encoder_25/dense/Tensordot/free:output:04transformer_encoder_25/dense/Tensordot/axes:output:0;transformer_encoder_25/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_encoder_25/dense/Tensordot/stackPack4transformer_encoder_25/dense/Tensordot/Prod:output:06transformer_encoder_25/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_encoder_25/dense/Tensordot/transpose	Transpose>transformer_encoder_25/layer_normalization/batchnorm/add_1:z:06transformer_encoder_25/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
.transformer_encoder_25/dense/Tensordot/ReshapeReshape4transformer_encoder_25/dense/Tensordot/transpose:y:05transformer_encoder_25/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_encoder_25/dense/Tensordot/MatMulMatMul7transformer_encoder_25/dense/Tensordot/Reshape:output:0=transformer_encoder_25/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
.transformer_encoder_25/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@v
4transformer_encoder_25/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_25/dense/Tensordot/concat_1ConcatV28transformer_encoder_25/dense/Tensordot/GatherV2:output:07transformer_encoder_25/dense/Tensordot/Const_2:output:0=transformer_encoder_25/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_encoder_25/dense/TensordotReshape7transformer_encoder_25/dense/Tensordot/MatMul:product:08transformer_encoder_25/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? @?
3transformer_encoder_25/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_encoder_25_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
$transformer_encoder_25/dense/BiasAddBiasAdd/transformer_encoder_25/dense/Tensordot:output:0;transformer_encoder_25/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @?
!transformer_encoder_25/dense/ReluRelu-transformer_encoder_25/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
7transformer_encoder_25/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_encoder_25_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0w
-transformer_encoder_25/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_encoder_25/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_encoder_25/dense_1/Tensordot/ShapeShape/transformer_encoder_25/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_encoder_25/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_25/dense_1/Tensordot/GatherV2GatherV27transformer_encoder_25/dense_1/Tensordot/Shape:output:06transformer_encoder_25/dense_1/Tensordot/free:output:0?transformer_encoder_25/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_encoder_25/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_encoder_25/dense_1/Tensordot/GatherV2_1GatherV27transformer_encoder_25/dense_1/Tensordot/Shape:output:06transformer_encoder_25/dense_1/Tensordot/axes:output:0Atransformer_encoder_25/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_encoder_25/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_encoder_25/dense_1/Tensordot/ProdProd:transformer_encoder_25/dense_1/Tensordot/GatherV2:output:07transformer_encoder_25/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_encoder_25/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_encoder_25/dense_1/Tensordot/Prod_1Prod<transformer_encoder_25/dense_1/Tensordot/GatherV2_1:output:09transformer_encoder_25/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_encoder_25/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_encoder_25/dense_1/Tensordot/concatConcatV26transformer_encoder_25/dense_1/Tensordot/free:output:06transformer_encoder_25/dense_1/Tensordot/axes:output:0=transformer_encoder_25/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_encoder_25/dense_1/Tensordot/stackPack6transformer_encoder_25/dense_1/Tensordot/Prod:output:08transformer_encoder_25/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_encoder_25/dense_1/Tensordot/transpose	Transpose/transformer_encoder_25/dense/Relu:activations:08transformer_encoder_25/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? @?
0transformer_encoder_25/dense_1/Tensordot/ReshapeReshape6transformer_encoder_25/dense_1/Tensordot/transpose:y:07transformer_encoder_25/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_encoder_25/dense_1/Tensordot/MatMulMatMul9transformer_encoder_25/dense_1/Tensordot/Reshape:output:0?transformer_encoder_25/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? z
0transformer_encoder_25/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: x
6transformer_encoder_25/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_encoder_25/dense_1/Tensordot/concat_1ConcatV2:transformer_encoder_25/dense_1/Tensordot/GatherV2:output:09transformer_encoder_25/dense_1/Tensordot/Const_2:output:0?transformer_encoder_25/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_encoder_25/dense_1/TensordotReshape9transformer_encoder_25/dense_1/Tensordot/MatMul:product:0:transformer_encoder_25/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_encoder_25_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&transformer_encoder_25/dense_1/BiasAddBiasAdd1transformer_encoder_25/dense_1/Tensordot:output:0=transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  s
.transformer_encoder_25/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
,transformer_encoder_25/dropout_1/dropout/MulMul/transformer_encoder_25/dense_1/BiasAdd:output:07transformer_encoder_25/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  ?
.transformer_encoder_25/dropout_1/dropout/ShapeShape/transformer_encoder_25/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
Etransformer_encoder_25/dropout_1/dropout/random_uniform/RandomUniformRandomUniform7transformer_encoder_25/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
dtype0|
7transformer_encoder_25/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
5transformer_encoder_25/dropout_1/dropout/GreaterEqualGreaterEqualNtransformer_encoder_25/dropout_1/dropout/random_uniform/RandomUniform:output:0@transformer_encoder_25/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????  ?
-transformer_encoder_25/dropout_1/dropout/CastCast9transformer_encoder_25/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
.transformer_encoder_25/dropout_1/dropout/Mul_1Mul0transformer_encoder_25/dropout_1/dropout/Mul:z:01transformer_encoder_25/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
transformer_encoder_25/add_1AddV2>transformer_encoder_25/layer_normalization/batchnorm/add_1:z:02transformer_encoder_25/dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ?
Ktransformer_encoder_25/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_encoder_25/layer_normalization_1/moments/meanMean transformer_encoder_25/add_1:z:0Ttransformer_encoder_25/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_encoder_25/layer_normalization_1/moments/StopGradientStopGradientBtransformer_encoder_25/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_encoder_25/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_encoder_25/add_1:z:0Jtransformer_encoder_25/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Otransformer_encoder_25/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_encoder_25/layer_normalization_1/moments/varianceMeanJtransformer_encoder_25/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_encoder_25/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_encoder_25/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_encoder_25/layer_normalization_1/batchnorm/addAddV2Ftransformer_encoder_25/layer_normalization_1/moments/variance:output:0Etransformer_encoder_25/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_encoder_25/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_encoder_25/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_encoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_encoder_25/layer_normalization_1/batchnorm/mulMul@transformer_encoder_25/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
<transformer_encoder_25/layer_normalization_1/batchnorm/mul_1Mul transformer_encoder_25/add_1:z:0>transformer_encoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_encoder_25/layer_normalization_1/batchnorm/mul_2MulBtransformer_encoder_25/layer_normalization_1/moments/mean:output:0>transformer_encoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Etransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_encoder_25_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_encoder_25/layer_normalization_1/batchnorm/subSubMtransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_encoder_25/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_encoder_25/layer_normalization_1/batchnorm/add_1AddV2@transformer_encoder_25/layer_normalization_1/batchnorm/mul_1:z:0>transformer_encoder_25/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
transformer_decoder_25/ShapeShape@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:t
*transformer_decoder_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,transformer_decoder_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,transformer_decoder_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$transformer_decoder_25/strided_sliceStridedSlice%transformer_decoder_25/Shape:output:03transformer_decoder_25/strided_slice/stack:output:05transformer_decoder_25/strided_slice/stack_1:output:05transformer_decoder_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_25/strided_slice_1StridedSlice%transformer_decoder_25/Shape:output:05transformer_decoder_25/strided_slice_1/stack:output:07transformer_decoder_25/strided_slice_1/stack_1:output:07transformer_decoder_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"transformer_decoder_25/range/startConst*
_output_shapes
: *
dtype0*
value	B : d
"transformer_decoder_25/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_25/rangeRange+transformer_decoder_25/range/start:output:0/transformer_decoder_25/strided_slice_1:output:0+transformer_decoder_25/range/delta:output:0*
_output_shapes
: }
,transformer_decoder_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
.transformer_decoder_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
&transformer_decoder_25/strided_slice_2StridedSlice%transformer_decoder_25/range:output:05transformer_decoder_25/strided_slice_2/stack:output:07transformer_decoder_25/strided_slice_2/stack_1:output:07transformer_decoder_25/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskf
$transformer_decoder_25/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : f
$transformer_decoder_25/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_25/range_1Range-transformer_decoder_25/range_1/start:output:0/transformer_decoder_25/strided_slice_1:output:0-transformer_decoder_25/range_1/delta:output:0*
_output_shapes
: ?
#transformer_decoder_25/GreaterEqualGreaterEqual/transformer_decoder_25/strided_slice_2:output:0'transformer_decoder_25/range_1:output:0*
T0*
_output_shapes

:  ?
transformer_decoder_25/CastCast'transformer_decoder_25/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  v
,transformer_decoder_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_25/strided_slice_3StridedSlice%transformer_decoder_25/Shape:output:05transformer_decoder_25/strided_slice_3/stack:output:07transformer_decoder_25/strided_slice_3/stack_1:output:07transformer_decoder_25/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
,transformer_decoder_25/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transformer_decoder_25/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&transformer_decoder_25/strided_slice_4StridedSlice%transformer_decoder_25/Shape:output:05transformer_decoder_25/strided_slice_4/stack:output:07transformer_decoder_25/strided_slice_4/stack_1:output:07transformer_decoder_25/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&transformer_decoder_25/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
$transformer_decoder_25/Reshape/shapePack/transformer_decoder_25/Reshape/shape/0:output:0/transformer_decoder_25/strided_slice_3:output:0/transformer_decoder_25/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_25/ReshapeReshapetransformer_decoder_25/Cast:y:0-transformer_decoder_25/Reshape/shape:output:0*
T0*"
_output_shapes
:  p
%transformer_decoder_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
!transformer_decoder_25/ExpandDims
ExpandDims-transformer_decoder_25/strided_slice:output:0.transformer_decoder_25/ExpandDims/dim:output:0*
T0*
_output_shapes
:m
transformer_decoder_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"      d
"transformer_decoder_25/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
transformer_decoder_25/concatConcatV2*transformer_decoder_25/ExpandDims:output:0%transformer_decoder_25/Const:output:0+transformer_decoder_25/concat/axis:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_25/TileTile'transformer_decoder_25/Reshape:output:0&transformer_decoder_25/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Ntransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_decoder_25/multi_head_attention/query/einsum/EinsumEinsum@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOpReadVariableOpMtransformer_decoder_25_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_25/multi_head_attention/query/addAddV2Htransformer_decoder_25/multi_head_attention/query/einsum/Einsum:output:0Ltransformer_decoder_25/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ltransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpUtransformer_decoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
=transformer_decoder_25/multi_head_attention/key/einsum/EinsumEinsum@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0Ttransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Btransformer_decoder_25/multi_head_attention/key/add/ReadVariableOpReadVariableOpKtransformer_decoder_25_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
3transformer_decoder_25/multi_head_attention/key/addAddV2Ftransformer_decoder_25/multi_head_attention/key/einsum/Einsum:output:0Jtransformer_decoder_25/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ntransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpWtransformer_decoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
?transformer_decoder_25/multi_head_attention/value/einsum/EinsumEinsum@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0Vtransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Dtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOpReadVariableOpMtransformer_decoder_25_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
5transformer_decoder_25/multi_head_attention/value/addAddV2Htransformer_decoder_25/multi_head_attention/value/einsum/Einsum:output:0Ltransformer_decoder_25/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? v
1transformer_decoder_25/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
/transformer_decoder_25/multi_head_attention/MulMul9transformer_decoder_25/multi_head_attention/query/add:z:0:transformer_decoder_25/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
9transformer_decoder_25/multi_head_attention/einsum/EinsumEinsum7transformer_decoder_25/multi_head_attention/key/add:z:03transformer_decoder_25/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
:transformer_decoder_25/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6transformer_decoder_25/multi_head_attention/ExpandDims
ExpandDims$transformer_decoder_25/Tile:output:0Ctransformer_decoder_25/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
8transformer_decoder_25/multi_head_attention/softmax/CastCast?transformer_decoder_25/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ~
9transformer_decoder_25/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7transformer_decoder_25/multi_head_attention/softmax/subSubBtransformer_decoder_25/multi_head_attention/softmax/sub/x:output:0<transformer_decoder_25/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ~
9transformer_decoder_25/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
7transformer_decoder_25/multi_head_attention/softmax/mulMul;transformer_decoder_25/multi_head_attention/softmax/sub:z:0Btransformer_decoder_25/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
7transformer_decoder_25/multi_head_attention/softmax/addAddV2Btransformer_decoder_25/multi_head_attention/einsum/Einsum:output:0;transformer_decoder_25/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_25/multi_head_attention/softmax/SoftmaxSoftmax;transformer_decoder_25/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_decoder_25/multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
Atransformer_decoder_25/multi_head_attention/dropout_2/dropout/MulMulEtransformer_decoder_25/multi_head_attention/softmax/Softmax:softmax:0Ltransformer_decoder_25/multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
Ctransformer_decoder_25/multi_head_attention/dropout_2/dropout/ShapeShapeEtransformer_decoder_25/multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Ztransformer_decoder_25/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniformLtransformer_decoder_25/multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0?
Ltransformer_decoder_25/multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Jtransformer_decoder_25/multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualctransformer_decoder_25/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0Utransformer_decoder_25/multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_decoder_25/multi_head_attention/dropout_2/dropout/CastCastNtransformer_decoder_25/multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
Ctransformer_decoder_25/multi_head_attention/dropout_2/dropout/Mul_1MulEtransformer_decoder_25/multi_head_attention/dropout_2/dropout/Mul:z:0Ftransformer_decoder_25/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
;transformer_decoder_25/multi_head_attention/einsum_1/EinsumEinsumGtransformer_decoder_25/multi_head_attention/dropout_2/dropout/Mul_1:z:09transformer_decoder_25/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Ytransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpbtransformer_decoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Jtransformer_decoder_25/multi_head_attention/attention_output/einsum/EinsumEinsumDtransformer_decoder_25/multi_head_attention/einsum_1/Einsum:output:0atransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
Otransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpXtransformer_decoder_25_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
@transformer_decoder_25/multi_head_attention/attention_output/addAddV2Stransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum:output:0Wtransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  q
,transformer_decoder_25/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
*transformer_decoder_25/dropout/dropout/MulMulDtransformer_decoder_25/multi_head_attention/attention_output/add:z:05transformer_decoder_25/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  ?
,transformer_decoder_25/dropout/dropout/ShapeShapeDtransformer_decoder_25/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
Ctransformer_decoder_25/dropout/dropout/random_uniform/RandomUniformRandomUniform5transformer_decoder_25/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
dtype0z
5transformer_decoder_25/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
3transformer_decoder_25/dropout/dropout/GreaterEqualGreaterEqualLtransformer_decoder_25/dropout/dropout/random_uniform/RandomUniform:output:0>transformer_decoder_25/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????  ?
+transformer_decoder_25/dropout/dropout/CastCast7transformer_decoder_25/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
,transformer_decoder_25/dropout/dropout/Mul_1Mul.transformer_decoder_25/dropout/dropout/Mul:z:0/transformer_decoder_25/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
transformer_decoder_25/addAddV20transformer_decoder_25/dropout/dropout/Mul_1:z:0@transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????  ?
Itransformer_decoder_25/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
7transformer_decoder_25/layer_normalization/moments/meanMeantransformer_decoder_25/add:z:0Rtransformer_decoder_25/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
?transformer_decoder_25/layer_normalization/moments/StopGradientStopGradient@transformer_decoder_25/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_decoder_25/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_25/add:z:0Htransformer_decoder_25/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Mtransformer_decoder_25/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
;transformer_decoder_25/layer_normalization/moments/varianceMeanHtransformer_decoder_25/layer_normalization/moments/SquaredDifference:z:0Vtransformer_decoder_25/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(
:transformer_decoder_25/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
8transformer_decoder_25/layer_normalization/batchnorm/addAddV2Dtransformer_decoder_25/layer_normalization/moments/variance:output:0Ctransformer_decoder_25/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
:transformer_decoder_25/layer_normalization/batchnorm/RsqrtRsqrt<transformer_decoder_25/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Gtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_decoder_25_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_decoder_25/layer_normalization/batchnorm/mulMul>transformer_decoder_25/layer_normalization/batchnorm/Rsqrt:y:0Otransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
:transformer_decoder_25/layer_normalization/batchnorm/mul_1Multransformer_decoder_25/add:z:0<transformer_decoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_decoder_25/layer_normalization/batchnorm/mul_2Mul@transformer_decoder_25/layer_normalization/moments/mean:output:0<transformer_decoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Ctransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpReadVariableOpLtransformer_decoder_25_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
8transformer_decoder_25/layer_normalization/batchnorm/subSubKtransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp:value:0>transformer_decoder_25/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
:transformer_decoder_25/layer_normalization/batchnorm/add_1AddV2>transformer_decoder_25/layer_normalization/batchnorm/mul_1:z:0<transformer_decoder_25/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
5transformer_decoder_25/dense/Tensordot/ReadVariableOpReadVariableOp>transformer_decoder_25_dense_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0u
+transformer_decoder_25/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:|
+transformer_decoder_25/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
,transformer_decoder_25/dense/Tensordot/ShapeShape>transformer_decoder_25/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:v
4transformer_decoder_25/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_25/dense/Tensordot/GatherV2GatherV25transformer_decoder_25/dense/Tensordot/Shape:output:04transformer_decoder_25/dense/Tensordot/free:output:0=transformer_decoder_25/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
6transformer_decoder_25/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_25/dense/Tensordot/GatherV2_1GatherV25transformer_decoder_25/dense/Tensordot/Shape:output:04transformer_decoder_25/dense/Tensordot/axes:output:0?transformer_decoder_25/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
,transformer_decoder_25/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
+transformer_decoder_25/dense/Tensordot/ProdProd8transformer_decoder_25/dense/Tensordot/GatherV2:output:05transformer_decoder_25/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: x
.transformer_decoder_25/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_25/dense/Tensordot/Prod_1Prod:transformer_decoder_25/dense/Tensordot/GatherV2_1:output:07transformer_decoder_25/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: t
2transformer_decoder_25/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-transformer_decoder_25/dense/Tensordot/concatConcatV24transformer_decoder_25/dense/Tensordot/free:output:04transformer_decoder_25/dense/Tensordot/axes:output:0;transformer_decoder_25/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
,transformer_decoder_25/dense/Tensordot/stackPack4transformer_decoder_25/dense/Tensordot/Prod:output:06transformer_decoder_25/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
0transformer_decoder_25/dense/Tensordot/transpose	Transpose>transformer_decoder_25/layer_normalization/batchnorm/add_1:z:06transformer_decoder_25/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
.transformer_decoder_25/dense/Tensordot/ReshapeReshape4transformer_decoder_25/dense/Tensordot/transpose:y:05transformer_decoder_25/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
-transformer_decoder_25/dense/Tensordot/MatMulMatMul7transformer_decoder_25/dense/Tensordot/Reshape:output:0=transformer_decoder_25/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@x
.transformer_decoder_25/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@v
4transformer_decoder_25/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_25/dense/Tensordot/concat_1ConcatV28transformer_decoder_25/dense/Tensordot/GatherV2:output:07transformer_decoder_25/dense/Tensordot/Const_2:output:0=transformer_decoder_25/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
&transformer_decoder_25/dense/TensordotReshape7transformer_decoder_25/dense/Tensordot/MatMul:product:08transformer_decoder_25/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? @?
3transformer_decoder_25/dense/BiasAdd/ReadVariableOpReadVariableOp<transformer_decoder_25_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
$transformer_decoder_25/dense/BiasAddBiasAdd/transformer_decoder_25/dense/Tensordot:output:0;transformer_decoder_25/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @?
!transformer_decoder_25/dense/ReluRelu-transformer_decoder_25/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
7transformer_decoder_25/dense_1/Tensordot/ReadVariableOpReadVariableOp@transformer_decoder_25_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0w
-transformer_decoder_25/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
-transformer_decoder_25/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
.transformer_decoder_25/dense_1/Tensordot/ShapeShape/transformer_decoder_25/dense/Relu:activations:0*
T0*
_output_shapes
:x
6transformer_decoder_25/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_25/dense_1/Tensordot/GatherV2GatherV27transformer_decoder_25/dense_1/Tensordot/Shape:output:06transformer_decoder_25/dense_1/Tensordot/free:output:0?transformer_decoder_25/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:z
8transformer_decoder_25/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
3transformer_decoder_25/dense_1/Tensordot/GatherV2_1GatherV27transformer_decoder_25/dense_1/Tensordot/Shape:output:06transformer_decoder_25/dense_1/Tensordot/axes:output:0Atransformer_decoder_25/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:x
.transformer_decoder_25/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
-transformer_decoder_25/dense_1/Tensordot/ProdProd:transformer_decoder_25/dense_1/Tensordot/GatherV2:output:07transformer_decoder_25/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: z
0transformer_decoder_25/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
/transformer_decoder_25/dense_1/Tensordot/Prod_1Prod<transformer_decoder_25/dense_1/Tensordot/GatherV2_1:output:09transformer_decoder_25/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: v
4transformer_decoder_25/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/transformer_decoder_25/dense_1/Tensordot/concatConcatV26transformer_decoder_25/dense_1/Tensordot/free:output:06transformer_decoder_25/dense_1/Tensordot/axes:output:0=transformer_decoder_25/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
.transformer_decoder_25/dense_1/Tensordot/stackPack6transformer_decoder_25/dense_1/Tensordot/Prod:output:08transformer_decoder_25/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
2transformer_decoder_25/dense_1/Tensordot/transpose	Transpose/transformer_decoder_25/dense/Relu:activations:08transformer_decoder_25/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? @?
0transformer_decoder_25/dense_1/Tensordot/ReshapeReshape6transformer_decoder_25/dense_1/Tensordot/transpose:y:07transformer_decoder_25/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
/transformer_decoder_25/dense_1/Tensordot/MatMulMatMul9transformer_decoder_25/dense_1/Tensordot/Reshape:output:0?transformer_decoder_25/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? z
0transformer_decoder_25/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: x
6transformer_decoder_25/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
1transformer_decoder_25/dense_1/Tensordot/concat_1ConcatV2:transformer_decoder_25/dense_1/Tensordot/GatherV2:output:09transformer_decoder_25/dense_1/Tensordot/Const_2:output:0?transformer_decoder_25/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
(transformer_decoder_25/dense_1/TensordotReshape9transformer_decoder_25/dense_1/Tensordot/MatMul:product:0:transformer_decoder_25/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOpReadVariableOp>transformer_decoder_25_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&transformer_decoder_25/dense_1/BiasAddBiasAdd1transformer_decoder_25/dense_1/Tensordot:output:0=transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  s
.transformer_decoder_25/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
,transformer_decoder_25/dropout_1/dropout/MulMul/transformer_decoder_25/dense_1/BiasAdd:output:07transformer_decoder_25/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  ?
.transformer_decoder_25/dropout_1/dropout/ShapeShape/transformer_decoder_25/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
Etransformer_decoder_25/dropout_1/dropout/random_uniform/RandomUniformRandomUniform7transformer_decoder_25/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
dtype0|
7transformer_decoder_25/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
5transformer_decoder_25/dropout_1/dropout/GreaterEqualGreaterEqualNtransformer_decoder_25/dropout_1/dropout/random_uniform/RandomUniform:output:0@transformer_decoder_25/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????  ?
-transformer_decoder_25/dropout_1/dropout/CastCast9transformer_decoder_25/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
.transformer_decoder_25/dropout_1/dropout/Mul_1Mul0transformer_decoder_25/dropout_1/dropout/Mul:z:01transformer_decoder_25/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
transformer_decoder_25/add_1AddV2>transformer_decoder_25/layer_normalization/batchnorm/add_1:z:02transformer_decoder_25/dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ?
Ktransformer_decoder_25/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
9transformer_decoder_25/layer_normalization_1/moments/meanMean transformer_decoder_25/add_1:z:0Ttransformer_decoder_25/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Atransformer_decoder_25/layer_normalization_1/moments/StopGradientStopGradientBtransformer_decoder_25/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_decoder_25/layer_normalization_1/moments/SquaredDifferenceSquaredDifference transformer_decoder_25/add_1:z:0Jtransformer_decoder_25/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Otransformer_decoder_25/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
=transformer_decoder_25/layer_normalization_1/moments/varianceMeanJtransformer_decoder_25/layer_normalization_1/moments/SquaredDifference:z:0Xtransformer_decoder_25/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
<transformer_decoder_25/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
:transformer_decoder_25/layer_normalization_1/batchnorm/addAddV2Ftransformer_decoder_25/layer_normalization_1/moments/variance:output:0Etransformer_decoder_25/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
<transformer_decoder_25/layer_normalization_1/batchnorm/RsqrtRsqrt>transformer_decoder_25/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Itransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpRtransformer_decoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_decoder_25/layer_normalization_1/batchnorm/mulMul@transformer_decoder_25/layer_normalization_1/batchnorm/Rsqrt:y:0Qtransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
<transformer_decoder_25/layer_normalization_1/batchnorm/mul_1Mul transformer_decoder_25/add_1:z:0>transformer_decoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_decoder_25/layer_normalization_1/batchnorm/mul_2MulBtransformer_decoder_25/layer_normalization_1/moments/mean:output:0>transformer_decoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Etransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpNtransformer_decoder_25_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
:transformer_decoder_25/layer_normalization_1/batchnorm/subSubMtransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp:value:0@transformer_decoder_25/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
<transformer_decoder_25/layer_normalization_1/batchnorm/add_1AddV2@transformer_decoder_25/layer_normalization_1/batchnorm/mul_1:z:0>transformer_decoder_25/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  s
1global_average_pooling1d_9/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_9/MeanMean@transformer_decoder_25/layer_normalization_1/batchnorm/add_1:z:0:global_average_pooling1d_9/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
dense_14/MatMulMatMul(global_average_pooling1d_9/Mean:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? h
dense_14/SoftmaxSoftmaxdense_14/BiasAdd:output:0*
T0*'
_output_shapes
:????????? i
IdentityIdentitydense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOpC^text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2D^token_and_position_embedding_23/position_embedding24/ReadVariableOpC^token_and_position_embedding_23/token_embedding24/embedding_lookupD^token_and_position_embedding_24/position_embedding25/ReadVariableOpC^token_and_position_embedding_24/token_embedding25/embedding_lookup4^transformer_decoder_25/dense/BiasAdd/ReadVariableOp6^transformer_decoder_25/dense/Tensordot/ReadVariableOp6^transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp8^transformer_decoder_25/dense_1/Tensordot/ReadVariableOpD^transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpH^transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_decoder_25/multi_head_attention/key/add/ReadVariableOpM^transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_decoder_25/multi_head_attention/query/add/ReadVariableOpO^transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_decoder_25/multi_head_attention/value/add/ReadVariableOpO^transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp4^transformer_encoder_25/dense/BiasAdd/ReadVariableOp6^transformer_encoder_25/dense/Tensordot/ReadVariableOp6^transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp8^transformer_encoder_25/dense_1/Tensordot/ReadVariableOpD^transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpH^transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpF^transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpJ^transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpP^transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpZ^transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpC^transformer_encoder_25/multi_head_attention/key/add/ReadVariableOpM^transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpE^transformer_encoder_25/multi_head_attention/query/add/ReadVariableOpO^transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpE^transformer_encoder_25/multi_head_attention/value/add/ReadVariableOpO^transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2?
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22?
Ctoken_and_position_embedding_23/position_embedding24/ReadVariableOpCtoken_and_position_embedding_23/position_embedding24/ReadVariableOp2?
Btoken_and_position_embedding_23/token_embedding24/embedding_lookupBtoken_and_position_embedding_23/token_embedding24/embedding_lookup2?
Ctoken_and_position_embedding_24/position_embedding25/ReadVariableOpCtoken_and_position_embedding_24/position_embedding25/ReadVariableOp2?
Btoken_and_position_embedding_24/token_embedding25/embedding_lookupBtoken_and_position_embedding_24/token_embedding25/embedding_lookup2j
3transformer_decoder_25/dense/BiasAdd/ReadVariableOp3transformer_decoder_25/dense/BiasAdd/ReadVariableOp2n
5transformer_decoder_25/dense/Tensordot/ReadVariableOp5transformer_decoder_25/dense/Tensordot/ReadVariableOp2n
5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp5transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp2r
7transformer_decoder_25/dense_1/Tensordot/ReadVariableOp7transformer_decoder_25/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpCtransformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_decoder_25/multi_head_attention/key/add/ReadVariableOpBtransformer_decoder_25/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOpDtransformer_decoder_25/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOpDtransformer_decoder_25/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp2j
3transformer_encoder_25/dense/BiasAdd/ReadVariableOp3transformer_encoder_25/dense/BiasAdd/ReadVariableOp2n
5transformer_encoder_25/dense/Tensordot/ReadVariableOp5transformer_encoder_25/dense/Tensordot/ReadVariableOp2n
5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp5transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp2r
7transformer_encoder_25/dense_1/Tensordot/ReadVariableOp7transformer_encoder_25/dense_1/Tensordot/ReadVariableOp2?
Ctransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpCtransformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp2?
Gtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpGtransformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp2?
Etransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpEtransformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp2?
Itransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpItransformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Otransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpOtransformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp2?
Ytransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpYtransformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Btransformer_encoder_25/multi_head_attention/key/add/ReadVariableOpBtransformer_encoder_25/multi_head_attention/key/add/ReadVariableOp2?
Ltransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpLtransformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOpDtransformer_encoder_25/multi_head_attention/query/add/ReadVariableOp2?
Ntransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpNtransformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Dtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOpDtransformer_encoder_25/multi_head_attention/value/add/ReadVariableOp2?
Ntransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpNtransformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:Q M
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
??
?8
!__inference__wrapped_model_500663

phrase

token_role[
Wmodel_9_text_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle\
Xmodel_9_text_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	8
4model_9_text_vectorization_9_string_lookup_9_equal_y;
7model_9_text_vectorization_9_string_lookup_9_selectv2_t	d
Qmodel_9_token_and_position_embedding_23_token_embedding24_embedding_lookup_500313:	? f
Tmodel_9_token_and_position_embedding_23_position_embedding24_readvariableop_resource:  c
Qmodel_9_token_and_position_embedding_24_token_embedding25_embedding_lookup_500337: f
Tmodel_9_token_and_position_embedding_24_position_embedding25_readvariableop_resource:  u
_model_9_transformer_encoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource: g
Umodel_9_transformer_encoder_25_multi_head_attention_query_add_readvariableop_resource:s
]model_9_transformer_encoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource: e
Smodel_9_transformer_encoder_25_multi_head_attention_key_add_readvariableop_resource:u
_model_9_transformer_encoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource: g
Umodel_9_transformer_encoder_25_multi_head_attention_value_add_readvariableop_resource:?
jmodel_9_transformer_encoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: n
`model_9_transformer_encoder_25_multi_head_attention_attention_output_add_readvariableop_resource: f
Xmodel_9_transformer_encoder_25_layer_normalization_batchnorm_mul_readvariableop_resource: b
Tmodel_9_transformer_encoder_25_layer_normalization_batchnorm_readvariableop_resource: X
Fmodel_9_transformer_encoder_25_dense_tensordot_readvariableop_resource: @R
Dmodel_9_transformer_encoder_25_dense_biasadd_readvariableop_resource:@Z
Hmodel_9_transformer_encoder_25_dense_1_tensordot_readvariableop_resource:@ T
Fmodel_9_transformer_encoder_25_dense_1_biasadd_readvariableop_resource: h
Zmodel_9_transformer_encoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource: d
Vmodel_9_transformer_encoder_25_layer_normalization_1_batchnorm_readvariableop_resource: u
_model_9_transformer_decoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource: g
Umodel_9_transformer_decoder_25_multi_head_attention_query_add_readvariableop_resource:s
]model_9_transformer_decoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource: e
Smodel_9_transformer_decoder_25_multi_head_attention_key_add_readvariableop_resource:u
_model_9_transformer_decoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource: g
Umodel_9_transformer_decoder_25_multi_head_attention_value_add_readvariableop_resource:?
jmodel_9_transformer_decoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource: n
`model_9_transformer_decoder_25_multi_head_attention_attention_output_add_readvariableop_resource: f
Xmodel_9_transformer_decoder_25_layer_normalization_batchnorm_mul_readvariableop_resource: b
Tmodel_9_transformer_decoder_25_layer_normalization_batchnorm_readvariableop_resource: X
Fmodel_9_transformer_decoder_25_dense_tensordot_readvariableop_resource: @R
Dmodel_9_transformer_decoder_25_dense_biasadd_readvariableop_resource:@Z
Hmodel_9_transformer_decoder_25_dense_1_tensordot_readvariableop_resource:@ T
Fmodel_9_transformer_decoder_25_dense_1_biasadd_readvariableop_resource: h
Zmodel_9_transformer_decoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource: d
Vmodel_9_transformer_decoder_25_layer_normalization_1_batchnorm_readvariableop_resource: A
/model_9_dense_14_matmul_readvariableop_resource:  >
0model_9_dense_14_biasadd_readvariableop_resource: 
identity??'model_9/dense_14/BiasAdd/ReadVariableOp?&model_9/dense_14/MatMul/ReadVariableOp?Jmodel_9/text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?Kmodel_9/token_and_position_embedding_23/position_embedding24/ReadVariableOp?Jmodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookup?Kmodel_9/token_and_position_embedding_24/position_embedding25/ReadVariableOp?Jmodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookup?;model_9/transformer_decoder_25/dense/BiasAdd/ReadVariableOp?=model_9/transformer_decoder_25/dense/Tensordot/ReadVariableOp?=model_9/transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp??model_9/transformer_decoder_25/dense_1/Tensordot/ReadVariableOp?Kmodel_9/transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp?Omodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp?Mmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp?Qmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp?Wmodel_9/transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp?amodel_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Jmodel_9/transformer_decoder_25/multi_head_attention/key/add/ReadVariableOp?Tmodel_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Lmodel_9/transformer_decoder_25/multi_head_attention/query/add/ReadVariableOp?Vmodel_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Lmodel_9/transformer_decoder_25/multi_head_attention/value/add/ReadVariableOp?Vmodel_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp?;model_9/transformer_encoder_25/dense/BiasAdd/ReadVariableOp?=model_9/transformer_encoder_25/dense/Tensordot/ReadVariableOp?=model_9/transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp??model_9/transformer_encoder_25/dense_1/Tensordot/ReadVariableOp?Kmodel_9/transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp?Omodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp?Mmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp?Qmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp?Wmodel_9/transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp?amodel_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Jmodel_9/transformer_encoder_25/multi_head_attention/key/add/ReadVariableOp?Tmodel_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Lmodel_9/transformer_encoder_25/multi_head_attention/query/add/ReadVariableOp?Vmodel_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Lmodel_9/transformer_encoder_25/multi_head_attention/value/add/ReadVariableOp?Vmodel_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
$model_9/text_vectorization_9/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????o
.model_9/text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
6model_9/text_vectorization_9/StringSplit/StringSplitV2StringSplitV2-model_9/text_vectorization_9/Squeeze:output:07model_9/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
<model_9/text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
>model_9/text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
>model_9/text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
6model_9/text_vectorization_9/StringSplit/strided_sliceStridedSlice@model_9/text_vectorization_9/StringSplit/StringSplitV2:indices:0Emodel_9/text_vectorization_9/StringSplit/strided_slice/stack:output:0Gmodel_9/text_vectorization_9/StringSplit/strided_slice/stack_1:output:0Gmodel_9/text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
>model_9/text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@model_9/text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@model_9/text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8model_9/text_vectorization_9/StringSplit/strided_slice_1StridedSlice>model_9/text_vectorization_9/StringSplit/StringSplitV2:shape:0Gmodel_9/text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Imodel_9/text_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Imodel_9/text_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
_model_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast?model_9/text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
amodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastAmodel_9/text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
imodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapecmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
imodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
hmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdrmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0rmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
mmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterqmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0vmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
hmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastomodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
gmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxcmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
imodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
gmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2pmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0rmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
gmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMullmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumemodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumemodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0omodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
qmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
kmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapecmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
lmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounttmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0omodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0tmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
fmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
amodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumsmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0omodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
jmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
fmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
amodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2smodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0gmodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0omodel_9/text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Jmodel_9/text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Wmodel_9_text_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle?model_9/text_vectorization_9/StringSplit/StringSplitV2:values:0Xmodel_9_text_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
2model_9/text_vectorization_9/string_lookup_9/EqualEqual?model_9/text_vectorization_9/StringSplit/StringSplitV2:values:04model_9_text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
5model_9/text_vectorization_9/string_lookup_9/SelectV2SelectV26model_9/text_vectorization_9/string_lookup_9/Equal:z:07model_9_text_vectorization_9_string_lookup_9_selectv2_tSmodel_9/text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
5model_9/text_vectorization_9/string_lookup_9/IdentityIdentity>model_9/text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????{
9model_9/text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
1model_9/text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
@model_9/text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor:model_9/text_vectorization_9/RaggedToTensor/Const:output:0>model_9/text_vectorization_9/string_lookup_9/Identity:output:0Bmodel_9/text_vectorization_9/RaggedToTensor/default_value:output:0Amodel_9/text_vectorization_9/StringSplit/strided_slice_1:output:0?model_9/text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Jmodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookupResourceGatherQmodel_9_token_and_position_embedding_23_token_embedding24_embedding_lookup_500313Imodel_9/text_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*d
_classZ
XVloc:@model_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/500313*+
_output_shapes
:?????????  *
dtype0?
Smodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/IdentityIdentitySmodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookup:output:0*
T0*d
_classZ
XVloc:@model_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/500313*+
_output_shapes
:?????????  ?
Umodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1Identity\model_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/token_and_position_embedding_23/position_embedding24/ShapeShape^model_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Pmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Rmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Rmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_9/token_and_position_embedding_23/position_embedding24/strided_sliceStridedSliceKmodel_9/token_and_position_embedding_23/position_embedding24/Shape:output:0Ymodel_9/token_and_position_embedding_23/position_embedding24/strided_slice/stack:output:0[model_9/token_and_position_embedding_23/position_embedding24/strided_slice/stack_1:output:0[model_9/token_and_position_embedding_23/position_embedding24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Kmodel_9/token_and_position_embedding_23/position_embedding24/ReadVariableOpReadVariableOpTmodel_9_token_and_position_embedding_23_position_embedding24_readvariableop_resource*
_output_shapes

:  *
dtype0?
Bmodel_9/token_and_position_embedding_23/position_embedding24/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_9/token_and_position_embedding_23/position_embedding24/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Tmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Rmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stackPackKmodel_9/token_and_position_embedding_23/position_embedding24/Const:output:0]model_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Vmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Tmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1PackSmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice:output:0_model_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Vmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Tmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2PackMmodel_9/token_and_position_embedding_23/position_embedding24/Const_1:output:0_model_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Lmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1StridedSliceSmodel_9/token_and_position_embedding_23/position_embedding24/ReadVariableOp:value:0[model_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack:output:0]model_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_1:output:0]model_9/token_and_position_embedding_23/position_embedding24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
Hmodel_9/token_and_position_embedding_23/position_embedding24/BroadcastToBroadcastToUmodel_9/token_and_position_embedding_23/position_embedding24/strided_slice_1:output:0Kmodel_9/token_and_position_embedding_23/position_embedding24/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
+model_9/token_and_position_embedding_23/addAddV2^model_9/token_and_position_embedding_23/token_embedding24/embedding_lookup/Identity_1:output:0Qmodel_9/token_and_position_embedding_23/position_embedding24/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  ?
>model_9/token_and_position_embedding_24/token_embedding25/CastCast
token_role*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
Jmodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookupResourceGatherQmodel_9_token_and_position_embedding_24_token_embedding25_embedding_lookup_500337Bmodel_9/token_and_position_embedding_24/token_embedding25/Cast:y:0*
Tindices0*d
_classZ
XVloc:@model_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/500337*+
_output_shapes
:?????????  *
dtype0?
Smodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/IdentityIdentitySmodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookup:output:0*
T0*d
_classZ
XVloc:@model_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/500337*+
_output_shapes
:?????????  ?
Umodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1Identity\model_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/token_and_position_embedding_24/position_embedding25/ShapeShape^model_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Pmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Rmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Rmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Jmodel_9/token_and_position_embedding_24/position_embedding25/strided_sliceStridedSliceKmodel_9/token_and_position_embedding_24/position_embedding25/Shape:output:0Ymodel_9/token_and_position_embedding_24/position_embedding25/strided_slice/stack:output:0[model_9/token_and_position_embedding_24/position_embedding25/strided_slice/stack_1:output:0[model_9/token_and_position_embedding_24/position_embedding25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Kmodel_9/token_and_position_embedding_24/position_embedding25/ReadVariableOpReadVariableOpTmodel_9_token_and_position_embedding_24_position_embedding25_readvariableop_resource*
_output_shapes

:  *
dtype0?
Bmodel_9/token_and_position_embedding_24/position_embedding25/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Dmodel_9/token_and_position_embedding_24/position_embedding25/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Tmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Rmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stackPackKmodel_9/token_and_position_embedding_24/position_embedding25/Const:output:0]model_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Vmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Tmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1PackSmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice:output:0_model_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Vmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Tmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2PackMmodel_9/token_and_position_embedding_24/position_embedding25/Const_1:output:0_model_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Lmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1StridedSliceSmodel_9/token_and_position_embedding_24/position_embedding25/ReadVariableOp:value:0[model_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack:output:0]model_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_1:output:0]model_9/token_and_position_embedding_24/position_embedding25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
Hmodel_9/token_and_position_embedding_24/position_embedding25/BroadcastToBroadcastToUmodel_9/token_and_position_embedding_24/position_embedding25/strided_slice_1:output:0Kmodel_9/token_and_position_embedding_24/position_embedding25/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
+model_9/token_and_position_embedding_24/addAddV2^model_9/token_and_position_embedding_24/token_embedding25/embedding_lookup/Identity_1:output:0Qmodel_9/token_and_position_embedding_24/position_embedding25/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  ?
model_9/add_9/addAddV2/model_9/token_and_position_embedding_23/add:z:0/model_9/token_and_position_embedding_24/add:z:0*
T0*+
_output_shapes
:?????????  ?
Vmodel_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp_model_9_transformer_encoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Gmodel_9/transformer_encoder_25/multi_head_attention/query/einsum/EinsumEinsummodel_9/add_9/add:z:0^model_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Lmodel_9/transformer_encoder_25/multi_head_attention/query/add/ReadVariableOpReadVariableOpUmodel_9_transformer_encoder_25_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
=model_9/transformer_encoder_25/multi_head_attention/query/addAddV2Pmodel_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum:output:0Tmodel_9/transformer_encoder_25/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Tmodel_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp]model_9_transformer_encoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Emodel_9/transformer_encoder_25/multi_head_attention/key/einsum/EinsumEinsummodel_9/add_9/add:z:0\model_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Jmodel_9/transformer_encoder_25/multi_head_attention/key/add/ReadVariableOpReadVariableOpSmodel_9_transformer_encoder_25_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
;model_9/transformer_encoder_25/multi_head_attention/key/addAddV2Nmodel_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum:output:0Rmodel_9/transformer_encoder_25/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Vmodel_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp_model_9_transformer_encoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Gmodel_9/transformer_encoder_25/multi_head_attention/value/einsum/EinsumEinsummodel_9/add_9/add:z:0^model_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Lmodel_9/transformer_encoder_25/multi_head_attention/value/add/ReadVariableOpReadVariableOpUmodel_9_transformer_encoder_25_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
=model_9/transformer_encoder_25/multi_head_attention/value/addAddV2Pmodel_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum:output:0Tmodel_9/transformer_encoder_25/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
9model_9/transformer_encoder_25/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
7model_9/transformer_encoder_25/multi_head_attention/MulMulAmodel_9/transformer_encoder_25/multi_head_attention/query/add:z:0Bmodel_9/transformer_encoder_25/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
Amodel_9/transformer_encoder_25/multi_head_attention/einsum/EinsumEinsum?model_9/transformer_encoder_25/multi_head_attention/key/add:z:0;model_9/transformer_encoder_25/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
Cmodel_9/transformer_encoder_25/multi_head_attention/softmax/SoftmaxSoftmaxJmodel_9/transformer_encoder_25/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
Fmodel_9/transformer_encoder_25/multi_head_attention/dropout_2/IdentityIdentityMmodel_9/transformer_encoder_25/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
Cmodel_9/transformer_encoder_25/multi_head_attention/einsum_1/EinsumEinsumOmodel_9/transformer_encoder_25/multi_head_attention/dropout_2/Identity:output:0Amodel_9/transformer_encoder_25/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
amodel_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpjmodel_9_transformer_encoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Rmodel_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/EinsumEinsumLmodel_9/transformer_encoder_25/multi_head_attention/einsum_1/Einsum:output:0imodel_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
Wmodel_9/transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp`model_9_transformer_encoder_25_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
Hmodel_9/transformer_encoder_25/multi_head_attention/attention_output/addAddV2[model_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum:output:0_model_9/transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
/model_9/transformer_encoder_25/dropout/IdentityIdentityLmodel_9/transformer_encoder_25/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  ?
"model_9/transformer_encoder_25/addAddV2model_9/add_9/add:z:08model_9/transformer_encoder_25/dropout/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Qmodel_9/transformer_encoder_25/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
?model_9/transformer_encoder_25/layer_normalization/moments/meanMean&model_9/transformer_encoder_25/add:z:0Zmodel_9/transformer_encoder_25/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Gmodel_9/transformer_encoder_25/layer_normalization/moments/StopGradientStopGradientHmodel_9/transformer_encoder_25/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Lmodel_9/transformer_encoder_25/layer_normalization/moments/SquaredDifferenceSquaredDifference&model_9/transformer_encoder_25/add:z:0Pmodel_9/transformer_encoder_25/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Umodel_9/transformer_encoder_25/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Cmodel_9/transformer_encoder_25/layer_normalization/moments/varianceMeanPmodel_9/transformer_encoder_25/layer_normalization/moments/SquaredDifference:z:0^model_9/transformer_encoder_25/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Bmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
@model_9/transformer_encoder_25/layer_normalization/batchnorm/addAddV2Lmodel_9/transformer_encoder_25/layer_normalization/moments/variance:output:0Kmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Bmodel_9/transformer_encoder_25/layer_normalization/batchnorm/RsqrtRsqrtDmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Omodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_9_transformer_encoder_25_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
@model_9/transformer_encoder_25/layer_normalization/batchnorm/mulMulFmodel_9/transformer_encoder_25/layer_normalization/batchnorm/Rsqrt:y:0Wmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul_1Mul&model_9/transformer_encoder_25/add:z:0Dmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul_2MulHmodel_9/transformer_encoder_25/layer_normalization/moments/mean:output:0Dmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Kmodel_9/transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpReadVariableOpTmodel_9_transformer_encoder_25_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
@model_9/transformer_encoder_25/layer_normalization/batchnorm/subSubSmodel_9/transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp:value:0Fmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add_1AddV2Fmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul_1:z:0Dmodel_9/transformer_encoder_25/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
=model_9/transformer_encoder_25/dense/Tensordot/ReadVariableOpReadVariableOpFmodel_9_transformer_encoder_25_dense_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0}
3model_9/transformer_encoder_25/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
3model_9/transformer_encoder_25/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
4model_9/transformer_encoder_25/dense/Tensordot/ShapeShapeFmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:~
<model_9/transformer_encoder_25/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7model_9/transformer_encoder_25/dense/Tensordot/GatherV2GatherV2=model_9/transformer_encoder_25/dense/Tensordot/Shape:output:0<model_9/transformer_encoder_25/dense/Tensordot/free:output:0Emodel_9/transformer_encoder_25/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
>model_9/transformer_encoder_25/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9model_9/transformer_encoder_25/dense/Tensordot/GatherV2_1GatherV2=model_9/transformer_encoder_25/dense/Tensordot/Shape:output:0<model_9/transformer_encoder_25/dense/Tensordot/axes:output:0Gmodel_9/transformer_encoder_25/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4model_9/transformer_encoder_25/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
3model_9/transformer_encoder_25/dense/Tensordot/ProdProd@model_9/transformer_encoder_25/dense/Tensordot/GatherV2:output:0=model_9/transformer_encoder_25/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
6model_9/transformer_encoder_25/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
5model_9/transformer_encoder_25/dense/Tensordot/Prod_1ProdBmodel_9/transformer_encoder_25/dense/Tensordot/GatherV2_1:output:0?model_9/transformer_encoder_25/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:model_9/transformer_encoder_25/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5model_9/transformer_encoder_25/dense/Tensordot/concatConcatV2<model_9/transformer_encoder_25/dense/Tensordot/free:output:0<model_9/transformer_encoder_25/dense/Tensordot/axes:output:0Cmodel_9/transformer_encoder_25/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
4model_9/transformer_encoder_25/dense/Tensordot/stackPack<model_9/transformer_encoder_25/dense/Tensordot/Prod:output:0>model_9/transformer_encoder_25/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
8model_9/transformer_encoder_25/dense/Tensordot/transpose	TransposeFmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add_1:z:0>model_9/transformer_encoder_25/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
6model_9/transformer_encoder_25/dense/Tensordot/ReshapeReshape<model_9/transformer_encoder_25/dense/Tensordot/transpose:y:0=model_9/transformer_encoder_25/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
5model_9/transformer_encoder_25/dense/Tensordot/MatMulMatMul?model_9/transformer_encoder_25/dense/Tensordot/Reshape:output:0Emodel_9/transformer_encoder_25/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
6model_9/transformer_encoder_25/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@~
<model_9/transformer_encoder_25/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7model_9/transformer_encoder_25/dense/Tensordot/concat_1ConcatV2@model_9/transformer_encoder_25/dense/Tensordot/GatherV2:output:0?model_9/transformer_encoder_25/dense/Tensordot/Const_2:output:0Emodel_9/transformer_encoder_25/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
.model_9/transformer_encoder_25/dense/TensordotReshape?model_9/transformer_encoder_25/dense/Tensordot/MatMul:product:0@model_9/transformer_encoder_25/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? @?
;model_9/transformer_encoder_25/dense/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_transformer_encoder_25_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
,model_9/transformer_encoder_25/dense/BiasAddBiasAdd7model_9/transformer_encoder_25/dense/Tensordot:output:0Cmodel_9/transformer_encoder_25/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @?
)model_9/transformer_encoder_25/dense/ReluRelu5model_9/transformer_encoder_25/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
?model_9/transformer_encoder_25/dense_1/Tensordot/ReadVariableOpReadVariableOpHmodel_9_transformer_encoder_25_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0
5model_9/transformer_encoder_25/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
5model_9/transformer_encoder_25/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
6model_9/transformer_encoder_25/dense_1/Tensordot/ShapeShape7model_9/transformer_encoder_25/dense/Relu:activations:0*
T0*
_output_shapes
:?
>model_9/transformer_encoder_25/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9model_9/transformer_encoder_25/dense_1/Tensordot/GatherV2GatherV2?model_9/transformer_encoder_25/dense_1/Tensordot/Shape:output:0>model_9/transformer_encoder_25/dense_1/Tensordot/free:output:0Gmodel_9/transformer_encoder_25/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
@model_9/transformer_encoder_25/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;model_9/transformer_encoder_25/dense_1/Tensordot/GatherV2_1GatherV2?model_9/transformer_encoder_25/dense_1/Tensordot/Shape:output:0>model_9/transformer_encoder_25/dense_1/Tensordot/axes:output:0Imodel_9/transformer_encoder_25/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
6model_9/transformer_encoder_25/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5model_9/transformer_encoder_25/dense_1/Tensordot/ProdProdBmodel_9/transformer_encoder_25/dense_1/Tensordot/GatherV2:output:0?model_9/transformer_encoder_25/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
8model_9/transformer_encoder_25/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
7model_9/transformer_encoder_25/dense_1/Tensordot/Prod_1ProdDmodel_9/transformer_encoder_25/dense_1/Tensordot/GatherV2_1:output:0Amodel_9/transformer_encoder_25/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<model_9/transformer_encoder_25/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7model_9/transformer_encoder_25/dense_1/Tensordot/concatConcatV2>model_9/transformer_encoder_25/dense_1/Tensordot/free:output:0>model_9/transformer_encoder_25/dense_1/Tensordot/axes:output:0Emodel_9/transformer_encoder_25/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6model_9/transformer_encoder_25/dense_1/Tensordot/stackPack>model_9/transformer_encoder_25/dense_1/Tensordot/Prod:output:0@model_9/transformer_encoder_25/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
:model_9/transformer_encoder_25/dense_1/Tensordot/transpose	Transpose7model_9/transformer_encoder_25/dense/Relu:activations:0@model_9/transformer_encoder_25/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? @?
8model_9/transformer_encoder_25/dense_1/Tensordot/ReshapeReshape>model_9/transformer_encoder_25/dense_1/Tensordot/transpose:y:0?model_9/transformer_encoder_25/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
7model_9/transformer_encoder_25/dense_1/Tensordot/MatMulMatMulAmodel_9/transformer_encoder_25/dense_1/Tensordot/Reshape:output:0Gmodel_9/transformer_encoder_25/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
8model_9/transformer_encoder_25/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ?
>model_9/transformer_encoder_25/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9model_9/transformer_encoder_25/dense_1/Tensordot/concat_1ConcatV2Bmodel_9/transformer_encoder_25/dense_1/Tensordot/GatherV2:output:0Amodel_9/transformer_encoder_25/dense_1/Tensordot/Const_2:output:0Gmodel_9/transformer_encoder_25/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
0model_9/transformer_encoder_25/dense_1/TensordotReshapeAmodel_9/transformer_encoder_25/dense_1/Tensordot/MatMul:product:0Bmodel_9/transformer_encoder_25/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
=model_9/transformer_encoder_25/dense_1/BiasAdd/ReadVariableOpReadVariableOpFmodel_9_transformer_encoder_25_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
.model_9/transformer_encoder_25/dense_1/BiasAddBiasAdd9model_9/transformer_encoder_25/dense_1/Tensordot:output:0Emodel_9/transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
1model_9/transformer_encoder_25/dropout_1/IdentityIdentity7model_9/transformer_encoder_25/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
$model_9/transformer_encoder_25/add_1AddV2Fmodel_9/transformer_encoder_25/layer_normalization/batchnorm/add_1:z:0:model_9/transformer_encoder_25/dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Smodel_9/transformer_encoder_25/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Amodel_9/transformer_encoder_25/layer_normalization_1/moments/meanMean(model_9/transformer_encoder_25/add_1:z:0\model_9/transformer_encoder_25/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Imodel_9/transformer_encoder_25/layer_normalization_1/moments/StopGradientStopGradientJmodel_9/transformer_encoder_25/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Nmodel_9/transformer_encoder_25/layer_normalization_1/moments/SquaredDifferenceSquaredDifference(model_9/transformer_encoder_25/add_1:z:0Rmodel_9/transformer_encoder_25/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Wmodel_9/transformer_encoder_25/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Emodel_9/transformer_encoder_25/layer_normalization_1/moments/varianceMeanRmodel_9/transformer_encoder_25/layer_normalization_1/moments/SquaredDifference:z:0`model_9/transformer_encoder_25/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Dmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Bmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/addAddV2Nmodel_9/transformer_encoder_25/layer_normalization_1/moments/variance:output:0Mmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Dmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/RsqrtRsqrtFmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Qmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpZmodel_9_transformer_encoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
Bmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mulMulHmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/Rsqrt:y:0Ymodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
Dmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul_1Mul(model_9/transformer_encoder_25/add_1:z:0Fmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Dmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul_2MulJmodel_9/transformer_encoder_25/layer_normalization_1/moments/mean:output:0Fmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Mmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpVmodel_9_transformer_encoder_25_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
Bmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/subSubUmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp:value:0Hmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
Dmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add_1AddV2Hmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul_1:z:0Fmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
$model_9/transformer_decoder_25/ShapeShapeHmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:|
2model_9/transformer_decoder_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4model_9/transformer_decoder_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4model_9/transformer_decoder_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,model_9/transformer_decoder_25/strided_sliceStridedSlice-model_9/transformer_decoder_25/Shape:output:0;model_9/transformer_decoder_25/strided_slice/stack:output:0=model_9/transformer_decoder_25/strided_slice/stack_1:output:0=model_9/transformer_decoder_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
4model_9/transformer_decoder_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6model_9/transformer_decoder_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model_9/transformer_decoder_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model_9/transformer_decoder_25/strided_slice_1StridedSlice-model_9/transformer_decoder_25/Shape:output:0=model_9/transformer_decoder_25/strided_slice_1/stack:output:0?model_9/transformer_decoder_25/strided_slice_1/stack_1:output:0?model_9/transformer_decoder_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*model_9/transformer_decoder_25/range/startConst*
_output_shapes
: *
dtype0*
value	B : l
*model_9/transformer_decoder_25/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
$model_9/transformer_decoder_25/rangeRange3model_9/transformer_decoder_25/range/start:output:07model_9/transformer_decoder_25/strided_slice_1:output:03model_9/transformer_decoder_25/range/delta:output:0*
_output_shapes
: ?
4model_9/transformer_decoder_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6model_9/transformer_decoder_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
6model_9/transformer_decoder_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.model_9/transformer_decoder_25/strided_slice_2StridedSlice-model_9/transformer_decoder_25/range:output:0=model_9/transformer_decoder_25/strided_slice_2/stack:output:0?model_9/transformer_decoder_25/strided_slice_2/stack_1:output:0?model_9/transformer_decoder_25/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskn
,model_9/transformer_decoder_25/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : n
,model_9/transformer_decoder_25/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
&model_9/transformer_decoder_25/range_1Range5model_9/transformer_decoder_25/range_1/start:output:07model_9/transformer_decoder_25/strided_slice_1:output:05model_9/transformer_decoder_25/range_1/delta:output:0*
_output_shapes
: ?
+model_9/transformer_decoder_25/GreaterEqualGreaterEqual7model_9/transformer_decoder_25/strided_slice_2:output:0/model_9/transformer_decoder_25/range_1:output:0*
T0*
_output_shapes

:  ?
#model_9/transformer_decoder_25/CastCast/model_9/transformer_decoder_25/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  ~
4model_9/transformer_decoder_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6model_9/transformer_decoder_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model_9/transformer_decoder_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model_9/transformer_decoder_25/strided_slice_3StridedSlice-model_9/transformer_decoder_25/Shape:output:0=model_9/transformer_decoder_25/strided_slice_3/stack:output:0?model_9/transformer_decoder_25/strided_slice_3/stack_1:output:0?model_9/transformer_decoder_25/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
4model_9/transformer_decoder_25/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:?
6model_9/transformer_decoder_25/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6model_9/transformer_decoder_25/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.model_9/transformer_decoder_25/strided_slice_4StridedSlice-model_9/transformer_decoder_25/Shape:output:0=model_9/transformer_decoder_25/strided_slice_4/stack:output:0?model_9/transformer_decoder_25/strided_slice_4/stack_1:output:0?model_9/transformer_decoder_25/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.model_9/transformer_decoder_25/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
,model_9/transformer_decoder_25/Reshape/shapePack7model_9/transformer_decoder_25/Reshape/shape/0:output:07model_9/transformer_decoder_25/strided_slice_3:output:07model_9/transformer_decoder_25/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
&model_9/transformer_decoder_25/ReshapeReshape'model_9/transformer_decoder_25/Cast:y:05model_9/transformer_decoder_25/Reshape/shape:output:0*
T0*"
_output_shapes
:  x
-model_9/transformer_decoder_25/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)model_9/transformer_decoder_25/ExpandDims
ExpandDims5model_9/transformer_decoder_25/strided_slice:output:06model_9/transformer_decoder_25/ExpandDims/dim:output:0*
T0*
_output_shapes
:u
$model_9/transformer_decoder_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"      l
*model_9/transformer_decoder_25/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%model_9/transformer_decoder_25/concatConcatV22model_9/transformer_decoder_25/ExpandDims:output:0-model_9/transformer_decoder_25/Const:output:03model_9/transformer_decoder_25/concat/axis:output:0*
N*
T0*
_output_shapes
:?
#model_9/transformer_decoder_25/TileTile/model_9/transformer_decoder_25/Reshape:output:0.model_9/transformer_decoder_25/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Vmodel_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp_model_9_transformer_decoder_25_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Gmodel_9/transformer_decoder_25/multi_head_attention/query/einsum/EinsumEinsumHmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0^model_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Lmodel_9/transformer_decoder_25/multi_head_attention/query/add/ReadVariableOpReadVariableOpUmodel_9_transformer_decoder_25_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
=model_9/transformer_decoder_25/multi_head_attention/query/addAddV2Pmodel_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum:output:0Tmodel_9/transformer_decoder_25/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Tmodel_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp]model_9_transformer_decoder_25_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Emodel_9/transformer_decoder_25/multi_head_attention/key/einsum/EinsumEinsumHmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0\model_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Jmodel_9/transformer_decoder_25/multi_head_attention/key/add/ReadVariableOpReadVariableOpSmodel_9_transformer_decoder_25_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
;model_9/transformer_decoder_25/multi_head_attention/key/addAddV2Nmodel_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum:output:0Rmodel_9/transformer_decoder_25/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Vmodel_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp_model_9_transformer_decoder_25_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Gmodel_9/transformer_decoder_25/multi_head_attention/value/einsum/EinsumEinsumHmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0^model_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Lmodel_9/transformer_decoder_25/multi_head_attention/value/add/ReadVariableOpReadVariableOpUmodel_9_transformer_decoder_25_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
=model_9/transformer_decoder_25/multi_head_attention/value/addAddV2Pmodel_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum:output:0Tmodel_9/transformer_decoder_25/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
9model_9/transformer_decoder_25/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
7model_9/transformer_decoder_25/multi_head_attention/MulMulAmodel_9/transformer_decoder_25/multi_head_attention/query/add:z:0Bmodel_9/transformer_decoder_25/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
Amodel_9/transformer_decoder_25/multi_head_attention/einsum/EinsumEinsum?model_9/transformer_decoder_25/multi_head_attention/key/add:z:0;model_9/transformer_decoder_25/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
Bmodel_9/transformer_decoder_25/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
>model_9/transformer_decoder_25/multi_head_attention/ExpandDims
ExpandDims,model_9/transformer_decoder_25/Tile:output:0Kmodel_9/transformer_decoder_25/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
@model_9/transformer_decoder_25/multi_head_attention/softmax/CastCastGmodel_9/transformer_decoder_25/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ?
Amodel_9/transformer_decoder_25/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
?model_9/transformer_decoder_25/multi_head_attention/softmax/subSubJmodel_9/transformer_decoder_25/multi_head_attention/softmax/sub/x:output:0Dmodel_9/transformer_decoder_25/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
Amodel_9/transformer_decoder_25/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
?model_9/transformer_decoder_25/multi_head_attention/softmax/mulMulCmodel_9/transformer_decoder_25/multi_head_attention/softmax/sub:z:0Jmodel_9/transformer_decoder_25/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
?model_9/transformer_decoder_25/multi_head_attention/softmax/addAddV2Jmodel_9/transformer_decoder_25/multi_head_attention/einsum/Einsum:output:0Cmodel_9/transformer_decoder_25/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
Cmodel_9/transformer_decoder_25/multi_head_attention/softmax/SoftmaxSoftmaxCmodel_9/transformer_decoder_25/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
Fmodel_9/transformer_decoder_25/multi_head_attention/dropout_2/IdentityIdentityMmodel_9/transformer_decoder_25/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
Cmodel_9/transformer_decoder_25/multi_head_attention/einsum_1/EinsumEinsumOmodel_9/transformer_decoder_25/multi_head_attention/dropout_2/Identity:output:0Amodel_9/transformer_decoder_25/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
amodel_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpjmodel_9_transformer_decoder_25_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
Rmodel_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/EinsumEinsumLmodel_9/transformer_decoder_25/multi_head_attention/einsum_1/Einsum:output:0imodel_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
Wmodel_9/transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp`model_9_transformer_decoder_25_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
Hmodel_9/transformer_decoder_25/multi_head_attention/attention_output/addAddV2[model_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum:output:0_model_9/transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
/model_9/transformer_decoder_25/dropout/IdentityIdentityLmodel_9/transformer_decoder_25/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  ?
"model_9/transformer_decoder_25/addAddV28model_9/transformer_decoder_25/dropout/Identity:output:0Hmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????  ?
Qmodel_9/transformer_decoder_25/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
?model_9/transformer_decoder_25/layer_normalization/moments/meanMean&model_9/transformer_decoder_25/add:z:0Zmodel_9/transformer_decoder_25/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Gmodel_9/transformer_decoder_25/layer_normalization/moments/StopGradientStopGradientHmodel_9/transformer_decoder_25/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Lmodel_9/transformer_decoder_25/layer_normalization/moments/SquaredDifferenceSquaredDifference&model_9/transformer_decoder_25/add:z:0Pmodel_9/transformer_decoder_25/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Umodel_9/transformer_decoder_25/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Cmodel_9/transformer_decoder_25/layer_normalization/moments/varianceMeanPmodel_9/transformer_decoder_25/layer_normalization/moments/SquaredDifference:z:0^model_9/transformer_decoder_25/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Bmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
@model_9/transformer_decoder_25/layer_normalization/batchnorm/addAddV2Lmodel_9/transformer_decoder_25/layer_normalization/moments/variance:output:0Kmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Bmodel_9/transformer_decoder_25/layer_normalization/batchnorm/RsqrtRsqrtDmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Omodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_9_transformer_decoder_25_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
@model_9/transformer_decoder_25/layer_normalization/batchnorm/mulMulFmodel_9/transformer_decoder_25/layer_normalization/batchnorm/Rsqrt:y:0Wmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul_1Mul&model_9/transformer_decoder_25/add:z:0Dmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul_2MulHmodel_9/transformer_decoder_25/layer_normalization/moments/mean:output:0Dmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Kmodel_9/transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpReadVariableOpTmodel_9_transformer_decoder_25_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
@model_9/transformer_decoder_25/layer_normalization/batchnorm/subSubSmodel_9/transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp:value:0Fmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
Bmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add_1AddV2Fmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul_1:z:0Dmodel_9/transformer_decoder_25/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
=model_9/transformer_decoder_25/dense/Tensordot/ReadVariableOpReadVariableOpFmodel_9_transformer_decoder_25_dense_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0}
3model_9/transformer_decoder_25/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
3model_9/transformer_decoder_25/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
4model_9/transformer_decoder_25/dense/Tensordot/ShapeShapeFmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:~
<model_9/transformer_decoder_25/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7model_9/transformer_decoder_25/dense/Tensordot/GatherV2GatherV2=model_9/transformer_decoder_25/dense/Tensordot/Shape:output:0<model_9/transformer_decoder_25/dense/Tensordot/free:output:0Emodel_9/transformer_decoder_25/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
>model_9/transformer_decoder_25/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9model_9/transformer_decoder_25/dense/Tensordot/GatherV2_1GatherV2=model_9/transformer_decoder_25/dense/Tensordot/Shape:output:0<model_9/transformer_decoder_25/dense/Tensordot/axes:output:0Gmodel_9/transformer_decoder_25/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4model_9/transformer_decoder_25/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
3model_9/transformer_decoder_25/dense/Tensordot/ProdProd@model_9/transformer_decoder_25/dense/Tensordot/GatherV2:output:0=model_9/transformer_decoder_25/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
6model_9/transformer_decoder_25/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
5model_9/transformer_decoder_25/dense/Tensordot/Prod_1ProdBmodel_9/transformer_decoder_25/dense/Tensordot/GatherV2_1:output:0?model_9/transformer_decoder_25/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:model_9/transformer_decoder_25/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
5model_9/transformer_decoder_25/dense/Tensordot/concatConcatV2<model_9/transformer_decoder_25/dense/Tensordot/free:output:0<model_9/transformer_decoder_25/dense/Tensordot/axes:output:0Cmodel_9/transformer_decoder_25/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
4model_9/transformer_decoder_25/dense/Tensordot/stackPack<model_9/transformer_decoder_25/dense/Tensordot/Prod:output:0>model_9/transformer_decoder_25/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
8model_9/transformer_decoder_25/dense/Tensordot/transpose	TransposeFmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add_1:z:0>model_9/transformer_decoder_25/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????  ?
6model_9/transformer_decoder_25/dense/Tensordot/ReshapeReshape<model_9/transformer_decoder_25/dense/Tensordot/transpose:y:0=model_9/transformer_decoder_25/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
5model_9/transformer_decoder_25/dense/Tensordot/MatMulMatMul?model_9/transformer_decoder_25/dense/Tensordot/Reshape:output:0Emodel_9/transformer_decoder_25/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
6model_9/transformer_decoder_25/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@~
<model_9/transformer_decoder_25/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7model_9/transformer_decoder_25/dense/Tensordot/concat_1ConcatV2@model_9/transformer_decoder_25/dense/Tensordot/GatherV2:output:0?model_9/transformer_decoder_25/dense/Tensordot/Const_2:output:0Emodel_9/transformer_decoder_25/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
.model_9/transformer_decoder_25/dense/TensordotReshape?model_9/transformer_decoder_25/dense/Tensordot/MatMul:product:0@model_9/transformer_decoder_25/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? @?
;model_9/transformer_decoder_25/dense/BiasAdd/ReadVariableOpReadVariableOpDmodel_9_transformer_decoder_25_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
,model_9/transformer_decoder_25/dense/BiasAddBiasAdd7model_9/transformer_decoder_25/dense/Tensordot:output:0Cmodel_9/transformer_decoder_25/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @?
)model_9/transformer_decoder_25/dense/ReluRelu5model_9/transformer_decoder_25/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
?model_9/transformer_decoder_25/dense_1/Tensordot/ReadVariableOpReadVariableOpHmodel_9_transformer_decoder_25_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0
5model_9/transformer_decoder_25/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
5model_9/transformer_decoder_25/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
6model_9/transformer_decoder_25/dense_1/Tensordot/ShapeShape7model_9/transformer_decoder_25/dense/Relu:activations:0*
T0*
_output_shapes
:?
>model_9/transformer_decoder_25/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9model_9/transformer_decoder_25/dense_1/Tensordot/GatherV2GatherV2?model_9/transformer_decoder_25/dense_1/Tensordot/Shape:output:0>model_9/transformer_decoder_25/dense_1/Tensordot/free:output:0Gmodel_9/transformer_decoder_25/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
@model_9/transformer_decoder_25/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;model_9/transformer_decoder_25/dense_1/Tensordot/GatherV2_1GatherV2?model_9/transformer_decoder_25/dense_1/Tensordot/Shape:output:0>model_9/transformer_decoder_25/dense_1/Tensordot/axes:output:0Imodel_9/transformer_decoder_25/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
6model_9/transformer_decoder_25/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5model_9/transformer_decoder_25/dense_1/Tensordot/ProdProdBmodel_9/transformer_decoder_25/dense_1/Tensordot/GatherV2:output:0?model_9/transformer_decoder_25/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
8model_9/transformer_decoder_25/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
7model_9/transformer_decoder_25/dense_1/Tensordot/Prod_1ProdDmodel_9/transformer_decoder_25/dense_1/Tensordot/GatherV2_1:output:0Amodel_9/transformer_decoder_25/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<model_9/transformer_decoder_25/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7model_9/transformer_decoder_25/dense_1/Tensordot/concatConcatV2>model_9/transformer_decoder_25/dense_1/Tensordot/free:output:0>model_9/transformer_decoder_25/dense_1/Tensordot/axes:output:0Emodel_9/transformer_decoder_25/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
6model_9/transformer_decoder_25/dense_1/Tensordot/stackPack>model_9/transformer_decoder_25/dense_1/Tensordot/Prod:output:0@model_9/transformer_decoder_25/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
:model_9/transformer_decoder_25/dense_1/Tensordot/transpose	Transpose7model_9/transformer_decoder_25/dense/Relu:activations:0@model_9/transformer_decoder_25/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? @?
8model_9/transformer_decoder_25/dense_1/Tensordot/ReshapeReshape>model_9/transformer_decoder_25/dense_1/Tensordot/transpose:y:0?model_9/transformer_decoder_25/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
7model_9/transformer_decoder_25/dense_1/Tensordot/MatMulMatMulAmodel_9/transformer_decoder_25/dense_1/Tensordot/Reshape:output:0Gmodel_9/transformer_decoder_25/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
8model_9/transformer_decoder_25/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: ?
>model_9/transformer_decoder_25/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9model_9/transformer_decoder_25/dense_1/Tensordot/concat_1ConcatV2Bmodel_9/transformer_decoder_25/dense_1/Tensordot/GatherV2:output:0Amodel_9/transformer_decoder_25/dense_1/Tensordot/Const_2:output:0Gmodel_9/transformer_decoder_25/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
0model_9/transformer_decoder_25/dense_1/TensordotReshapeAmodel_9/transformer_decoder_25/dense_1/Tensordot/MatMul:product:0Bmodel_9/transformer_decoder_25/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????  ?
=model_9/transformer_decoder_25/dense_1/BiasAdd/ReadVariableOpReadVariableOpFmodel_9_transformer_decoder_25_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
.model_9/transformer_decoder_25/dense_1/BiasAddBiasAdd9model_9/transformer_decoder_25/dense_1/Tensordot:output:0Emodel_9/transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
1model_9/transformer_decoder_25/dropout_1/IdentityIdentity7model_9/transformer_decoder_25/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
$model_9/transformer_decoder_25/add_1AddV2Fmodel_9/transformer_decoder_25/layer_normalization/batchnorm/add_1:z:0:model_9/transformer_decoder_25/dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
Smodel_9/transformer_decoder_25/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Amodel_9/transformer_decoder_25/layer_normalization_1/moments/meanMean(model_9/transformer_decoder_25/add_1:z:0\model_9/transformer_decoder_25/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Imodel_9/transformer_decoder_25/layer_normalization_1/moments/StopGradientStopGradientJmodel_9/transformer_decoder_25/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Nmodel_9/transformer_decoder_25/layer_normalization_1/moments/SquaredDifferenceSquaredDifference(model_9/transformer_decoder_25/add_1:z:0Rmodel_9/transformer_decoder_25/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????  ?
Wmodel_9/transformer_decoder_25/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Emodel_9/transformer_decoder_25/layer_normalization_1/moments/varianceMeanRmodel_9/transformer_decoder_25/layer_normalization_1/moments/SquaredDifference:z:0`model_9/transformer_decoder_25/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Dmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Bmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/addAddV2Nmodel_9/transformer_decoder_25/layer_normalization_1/moments/variance:output:0Mmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Dmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/RsqrtRsqrtFmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Qmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpZmodel_9_transformer_decoder_25_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0?
Bmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mulMulHmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/Rsqrt:y:0Ymodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
Dmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul_1Mul(model_9/transformer_decoder_25/add_1:z:0Fmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Dmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul_2MulJmodel_9/transformer_decoder_25/layer_normalization_1/moments/mean:output:0Fmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
Mmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpVmodel_9_transformer_decoder_25_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
Bmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/subSubUmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp:value:0Hmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
Dmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/add_1AddV2Hmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul_1:z:0Fmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  {
9model_9/global_average_pooling1d_9/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
'model_9/global_average_pooling1d_9/MeanMeanHmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/add_1:z:0Bmodel_9/global_average_pooling1d_9/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? ?
&model_9/dense_14/MatMul/ReadVariableOpReadVariableOp/model_9_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0?
model_9/dense_14/MatMulMatMul0model_9/global_average_pooling1d_9/Mean:output:0.model_9/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
'model_9/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_9_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_9/dense_14/BiasAddBiasAdd!model_9/dense_14/MatMul:product:0/model_9/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? x
model_9/dense_14/SoftmaxSoftmax!model_9/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:????????? q
IdentityIdentity"model_9/dense_14/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp(^model_9/dense_14/BiasAdd/ReadVariableOp'^model_9/dense_14/MatMul/ReadVariableOpK^model_9/text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2L^model_9/token_and_position_embedding_23/position_embedding24/ReadVariableOpK^model_9/token_and_position_embedding_23/token_embedding24/embedding_lookupL^model_9/token_and_position_embedding_24/position_embedding25/ReadVariableOpK^model_9/token_and_position_embedding_24/token_embedding25/embedding_lookup<^model_9/transformer_decoder_25/dense/BiasAdd/ReadVariableOp>^model_9/transformer_decoder_25/dense/Tensordot/ReadVariableOp>^model_9/transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp@^model_9/transformer_decoder_25/dense_1/Tensordot/ReadVariableOpL^model_9/transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpP^model_9/transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpN^model_9/transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpR^model_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpX^model_9/transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpb^model_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpK^model_9/transformer_decoder_25/multi_head_attention/key/add/ReadVariableOpU^model_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpM^model_9/transformer_decoder_25/multi_head_attention/query/add/ReadVariableOpW^model_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpM^model_9/transformer_decoder_25/multi_head_attention/value/add/ReadVariableOpW^model_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp<^model_9/transformer_encoder_25/dense/BiasAdd/ReadVariableOp>^model_9/transformer_encoder_25/dense/Tensordot/ReadVariableOp>^model_9/transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp@^model_9/transformer_encoder_25/dense_1/Tensordot/ReadVariableOpL^model_9/transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpP^model_9/transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpN^model_9/transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpR^model_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpX^model_9/transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpb^model_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpK^model_9/transformer_encoder_25/multi_head_attention/key/add/ReadVariableOpU^model_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpM^model_9/transformer_encoder_25/multi_head_attention/query/add/ReadVariableOpW^model_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpM^model_9/transformer_encoder_25/multi_head_attention/value/add/ReadVariableOpW^model_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'model_9/dense_14/BiasAdd/ReadVariableOp'model_9/dense_14/BiasAdd/ReadVariableOp2P
&model_9/dense_14/MatMul/ReadVariableOp&model_9/dense_14/MatMul/ReadVariableOp2?
Jmodel_9/text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Jmodel_9/text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22?
Kmodel_9/token_and_position_embedding_23/position_embedding24/ReadVariableOpKmodel_9/token_and_position_embedding_23/position_embedding24/ReadVariableOp2?
Jmodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookupJmodel_9/token_and_position_embedding_23/token_embedding24/embedding_lookup2?
Kmodel_9/token_and_position_embedding_24/position_embedding25/ReadVariableOpKmodel_9/token_and_position_embedding_24/position_embedding25/ReadVariableOp2?
Jmodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookupJmodel_9/token_and_position_embedding_24/token_embedding25/embedding_lookup2z
;model_9/transformer_decoder_25/dense/BiasAdd/ReadVariableOp;model_9/transformer_decoder_25/dense/BiasAdd/ReadVariableOp2~
=model_9/transformer_decoder_25/dense/Tensordot/ReadVariableOp=model_9/transformer_decoder_25/dense/Tensordot/ReadVariableOp2~
=model_9/transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp=model_9/transformer_decoder_25/dense_1/BiasAdd/ReadVariableOp2?
?model_9/transformer_decoder_25/dense_1/Tensordot/ReadVariableOp?model_9/transformer_decoder_25/dense_1/Tensordot/ReadVariableOp2?
Kmodel_9/transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOpKmodel_9/transformer_decoder_25/layer_normalization/batchnorm/ReadVariableOp2?
Omodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOpOmodel_9/transformer_decoder_25/layer_normalization/batchnorm/mul/ReadVariableOp2?
Mmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOpMmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/ReadVariableOp2?
Qmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpQmodel_9/transformer_decoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Wmodel_9/transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOpWmodel_9/transformer_decoder_25/multi_head_attention/attention_output/add/ReadVariableOp2?
amodel_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpamodel_9/transformer_decoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Jmodel_9/transformer_decoder_25/multi_head_attention/key/add/ReadVariableOpJmodel_9/transformer_decoder_25/multi_head_attention/key/add/ReadVariableOp2?
Tmodel_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpTmodel_9/transformer_decoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Lmodel_9/transformer_decoder_25/multi_head_attention/query/add/ReadVariableOpLmodel_9/transformer_decoder_25/multi_head_attention/query/add/ReadVariableOp2?
Vmodel_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpVmodel_9/transformer_decoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Lmodel_9/transformer_decoder_25/multi_head_attention/value/add/ReadVariableOpLmodel_9/transformer_decoder_25/multi_head_attention/value/add/ReadVariableOp2?
Vmodel_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpVmodel_9/transformer_decoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp2z
;model_9/transformer_encoder_25/dense/BiasAdd/ReadVariableOp;model_9/transformer_encoder_25/dense/BiasAdd/ReadVariableOp2~
=model_9/transformer_encoder_25/dense/Tensordot/ReadVariableOp=model_9/transformer_encoder_25/dense/Tensordot/ReadVariableOp2~
=model_9/transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp=model_9/transformer_encoder_25/dense_1/BiasAdd/ReadVariableOp2?
?model_9/transformer_encoder_25/dense_1/Tensordot/ReadVariableOp?model_9/transformer_encoder_25/dense_1/Tensordot/ReadVariableOp2?
Kmodel_9/transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOpKmodel_9/transformer_encoder_25/layer_normalization/batchnorm/ReadVariableOp2?
Omodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOpOmodel_9/transformer_encoder_25/layer_normalization/batchnorm/mul/ReadVariableOp2?
Mmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOpMmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/ReadVariableOp2?
Qmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOpQmodel_9/transformer_encoder_25/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Wmodel_9/transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOpWmodel_9/transformer_encoder_25/multi_head_attention/attention_output/add/ReadVariableOp2?
amodel_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpamodel_9/transformer_encoder_25/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Jmodel_9/transformer_encoder_25/multi_head_attention/key/add/ReadVariableOpJmodel_9/transformer_encoder_25/multi_head_attention/key/add/ReadVariableOp2?
Tmodel_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOpTmodel_9/transformer_encoder_25/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Lmodel_9/transformer_encoder_25/multi_head_attention/query/add/ReadVariableOpLmodel_9/transformer_encoder_25/multi_head_attention/query/add/ReadVariableOp2?
Vmodel_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOpVmodel_9/transformer_encoder_25/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Lmodel_9/transformer_encoder_25/multi_head_attention/value/add/ReadVariableOpLmodel_9/transformer_encoder_25/multi_head_attention/value/add/ReadVariableOp2?
Vmodel_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOpVmodel_9/transformer_encoder_25/multi_head_attention/value/einsum/Einsum/ReadVariableOp:O K
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
?
__inference__initializer_5045809
5key_value_init393025_lookuptableimportv2_table_handle1
-key_value_init393025_lookuptableimportv2_keys3
/key_value_init393025_lookuptableimportv2_values	
identity??(key_value_init393025/LookupTableImportV2?
(key_value_init393025/LookupTableImportV2LookupTableImportV25key_value_init393025_lookuptableimportv2_table_handle-key_value_init393025_lookuptableimportv2_keys/key_value_init393025_lookuptableimportv2_values*	
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
NoOpNoOp)^key_value_init393025/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2T
(key_value_init393025/LookupTableImportV2(key_value_init393025/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
??
?k
"__inference__traced_restore_505426
file_prefix2
 assignvariableop_dense_14_kernel:  .
 assignvariableop_1_dense_14_bias: b
Oassignvariableop_2_token_and_position_embedding_23_token_embedding24_embeddings:	? d
Rassignvariableop_3_token_and_position_embedding_23_position_embedding24_embeddings:  a
Oassignvariableop_4_token_and_position_embedding_24_token_embedding25_embeddings: d
Rassignvariableop_5_token_and_position_embedding_24_position_embedding25_embeddings:  a
Kassignvariableop_6_transformer_encoder_25_multi_head_attention_query_kernel: [
Iassignvariableop_7_transformer_encoder_25_multi_head_attention_query_bias:_
Iassignvariableop_8_transformer_encoder_25_multi_head_attention_key_kernel: Y
Gassignvariableop_9_transformer_encoder_25_multi_head_attention_key_bias:b
Lassignvariableop_10_transformer_encoder_25_multi_head_attention_value_kernel: \
Jassignvariableop_11_transformer_encoder_25_multi_head_attention_value_bias:m
Wassignvariableop_12_transformer_encoder_25_multi_head_attention_attention_output_kernel: c
Uassignvariableop_13_transformer_encoder_25_multi_head_attention_attention_output_bias: R
Dassignvariableop_14_transformer_encoder_25_layer_normalization_gamma: Q
Cassignvariableop_15_transformer_encoder_25_layer_normalization_beta: T
Fassignvariableop_16_transformer_encoder_25_layer_normalization_1_gamma: S
Eassignvariableop_17_transformer_encoder_25_layer_normalization_1_beta: I
7assignvariableop_18_transformer_encoder_25_dense_kernel: @C
5assignvariableop_19_transformer_encoder_25_dense_bias:@K
9assignvariableop_20_transformer_encoder_25_dense_1_kernel:@ E
7assignvariableop_21_transformer_encoder_25_dense_1_bias: b
Lassignvariableop_22_transformer_decoder_25_multi_head_attention_query_kernel: \
Jassignvariableop_23_transformer_decoder_25_multi_head_attention_query_bias:`
Jassignvariableop_24_transformer_decoder_25_multi_head_attention_key_kernel: Z
Hassignvariableop_25_transformer_decoder_25_multi_head_attention_key_bias:b
Lassignvariableop_26_transformer_decoder_25_multi_head_attention_value_kernel: \
Jassignvariableop_27_transformer_decoder_25_multi_head_attention_value_bias:m
Wassignvariableop_28_transformer_decoder_25_multi_head_attention_attention_output_kernel: c
Uassignvariableop_29_transformer_decoder_25_multi_head_attention_attention_output_bias: R
Dassignvariableop_30_transformer_decoder_25_layer_normalization_gamma: Q
Cassignvariableop_31_transformer_decoder_25_layer_normalization_beta: T
Fassignvariableop_32_transformer_decoder_25_layer_normalization_1_gamma: S
Eassignvariableop_33_transformer_decoder_25_layer_normalization_1_beta: I
7assignvariableop_34_transformer_decoder_25_dense_kernel: @C
5assignvariableop_35_transformer_decoder_25_dense_bias:@K
9assignvariableop_36_transformer_decoder_25_dense_1_kernel:@ E
7assignvariableop_37_transformer_decoder_25_dense_1_bias: '
assignvariableop_38_adam_iter:	 )
assignvariableop_39_adam_beta_1: )
assignvariableop_40_adam_beta_2: (
assignvariableop_41_adam_decay: 0
&assignvariableop_42_adam_learning_rate: 
mutablehashtable: %
assignvariableop_43_total_1: %
assignvariableop_44_count_1: #
assignvariableop_45_total: #
assignvariableop_46_count: <
*assignvariableop_47_adam_dense_14_kernel_m:  6
(assignvariableop_48_adam_dense_14_bias_m: j
Wassignvariableop_49_adam_token_and_position_embedding_23_token_embedding24_embeddings_m:	? l
Zassignvariableop_50_adam_token_and_position_embedding_23_position_embedding24_embeddings_m:  i
Wassignvariableop_51_adam_token_and_position_embedding_24_token_embedding25_embeddings_m: l
Zassignvariableop_52_adam_token_and_position_embedding_24_position_embedding25_embeddings_m:  i
Sassignvariableop_53_adam_transformer_encoder_25_multi_head_attention_query_kernel_m: c
Qassignvariableop_54_adam_transformer_encoder_25_multi_head_attention_query_bias_m:g
Qassignvariableop_55_adam_transformer_encoder_25_multi_head_attention_key_kernel_m: a
Oassignvariableop_56_adam_transformer_encoder_25_multi_head_attention_key_bias_m:i
Sassignvariableop_57_adam_transformer_encoder_25_multi_head_attention_value_kernel_m: c
Qassignvariableop_58_adam_transformer_encoder_25_multi_head_attention_value_bias_m:t
^assignvariableop_59_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_m: j
\assignvariableop_60_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_m: Y
Kassignvariableop_61_adam_transformer_encoder_25_layer_normalization_gamma_m: X
Jassignvariableop_62_adam_transformer_encoder_25_layer_normalization_beta_m: [
Massignvariableop_63_adam_transformer_encoder_25_layer_normalization_1_gamma_m: Z
Lassignvariableop_64_adam_transformer_encoder_25_layer_normalization_1_beta_m: P
>assignvariableop_65_adam_transformer_encoder_25_dense_kernel_m: @J
<assignvariableop_66_adam_transformer_encoder_25_dense_bias_m:@R
@assignvariableop_67_adam_transformer_encoder_25_dense_1_kernel_m:@ L
>assignvariableop_68_adam_transformer_encoder_25_dense_1_bias_m: i
Sassignvariableop_69_adam_transformer_decoder_25_multi_head_attention_query_kernel_m: c
Qassignvariableop_70_adam_transformer_decoder_25_multi_head_attention_query_bias_m:g
Qassignvariableop_71_adam_transformer_decoder_25_multi_head_attention_key_kernel_m: a
Oassignvariableop_72_adam_transformer_decoder_25_multi_head_attention_key_bias_m:i
Sassignvariableop_73_adam_transformer_decoder_25_multi_head_attention_value_kernel_m: c
Qassignvariableop_74_adam_transformer_decoder_25_multi_head_attention_value_bias_m:t
^assignvariableop_75_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_m: j
\assignvariableop_76_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_m: Y
Kassignvariableop_77_adam_transformer_decoder_25_layer_normalization_gamma_m: X
Jassignvariableop_78_adam_transformer_decoder_25_layer_normalization_beta_m: [
Massignvariableop_79_adam_transformer_decoder_25_layer_normalization_1_gamma_m: Z
Lassignvariableop_80_adam_transformer_decoder_25_layer_normalization_1_beta_m: P
>assignvariableop_81_adam_transformer_decoder_25_dense_kernel_m: @J
<assignvariableop_82_adam_transformer_decoder_25_dense_bias_m:@R
@assignvariableop_83_adam_transformer_decoder_25_dense_1_kernel_m:@ L
>assignvariableop_84_adam_transformer_decoder_25_dense_1_bias_m: <
*assignvariableop_85_adam_dense_14_kernel_v:  6
(assignvariableop_86_adam_dense_14_bias_v: j
Wassignvariableop_87_adam_token_and_position_embedding_23_token_embedding24_embeddings_v:	? l
Zassignvariableop_88_adam_token_and_position_embedding_23_position_embedding24_embeddings_v:  i
Wassignvariableop_89_adam_token_and_position_embedding_24_token_embedding25_embeddings_v: l
Zassignvariableop_90_adam_token_and_position_embedding_24_position_embedding25_embeddings_v:  i
Sassignvariableop_91_adam_transformer_encoder_25_multi_head_attention_query_kernel_v: c
Qassignvariableop_92_adam_transformer_encoder_25_multi_head_attention_query_bias_v:g
Qassignvariableop_93_adam_transformer_encoder_25_multi_head_attention_key_kernel_v: a
Oassignvariableop_94_adam_transformer_encoder_25_multi_head_attention_key_bias_v:i
Sassignvariableop_95_adam_transformer_encoder_25_multi_head_attention_value_kernel_v: c
Qassignvariableop_96_adam_transformer_encoder_25_multi_head_attention_value_bias_v:t
^assignvariableop_97_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_v: j
\assignvariableop_98_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_v: Y
Kassignvariableop_99_adam_transformer_encoder_25_layer_normalization_gamma_v: Y
Kassignvariableop_100_adam_transformer_encoder_25_layer_normalization_beta_v: \
Nassignvariableop_101_adam_transformer_encoder_25_layer_normalization_1_gamma_v: [
Massignvariableop_102_adam_transformer_encoder_25_layer_normalization_1_beta_v: Q
?assignvariableop_103_adam_transformer_encoder_25_dense_kernel_v: @K
=assignvariableop_104_adam_transformer_encoder_25_dense_bias_v:@S
Aassignvariableop_105_adam_transformer_encoder_25_dense_1_kernel_v:@ M
?assignvariableop_106_adam_transformer_encoder_25_dense_1_bias_v: j
Tassignvariableop_107_adam_transformer_decoder_25_multi_head_attention_query_kernel_v: d
Rassignvariableop_108_adam_transformer_decoder_25_multi_head_attention_query_bias_v:h
Rassignvariableop_109_adam_transformer_decoder_25_multi_head_attention_key_kernel_v: b
Passignvariableop_110_adam_transformer_decoder_25_multi_head_attention_key_bias_v:j
Tassignvariableop_111_adam_transformer_decoder_25_multi_head_attention_value_kernel_v: d
Rassignvariableop_112_adam_transformer_decoder_25_multi_head_attention_value_bias_v:u
_assignvariableop_113_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_v: k
]assignvariableop_114_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_v: Z
Lassignvariableop_115_adam_transformer_decoder_25_layer_normalization_gamma_v: Y
Kassignvariableop_116_adam_transformer_decoder_25_layer_normalization_beta_v: \
Nassignvariableop_117_adam_transformer_decoder_25_layer_normalization_1_gamma_v: [
Massignvariableop_118_adam_transformer_decoder_25_layer_normalization_1_beta_v: Q
?assignvariableop_119_adam_transformer_decoder_25_dense_kernel_v: @K
=assignvariableop_120_adam_transformer_decoder_25_dense_bias_v:@S
Aassignvariableop_121_adam_transformer_decoder_25_dense_1_kernel_v:@ M
?assignvariableop_122_adam_transformer_decoder_25_dense_1_bias_v: 
identity_124??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?StatefulPartitionedCall?;
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?:
value?:B?:~B6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?
value?B?~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2~		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpOassignvariableop_2_token_and_position_embedding_23_token_embedding24_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpRassignvariableop_3_token_and_position_embedding_23_position_embedding24_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpOassignvariableop_4_token_and_position_embedding_24_token_embedding25_embeddingsIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpRassignvariableop_5_token_and_position_embedding_24_position_embedding25_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpKassignvariableop_6_transformer_encoder_25_multi_head_attention_query_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpIassignvariableop_7_transformer_encoder_25_multi_head_attention_query_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpIassignvariableop_8_transformer_encoder_25_multi_head_attention_key_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpGassignvariableop_9_transformer_encoder_25_multi_head_attention_key_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpLassignvariableop_10_transformer_encoder_25_multi_head_attention_value_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpJassignvariableop_11_transformer_encoder_25_multi_head_attention_value_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpWassignvariableop_12_transformer_encoder_25_multi_head_attention_attention_output_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpUassignvariableop_13_transformer_encoder_25_multi_head_attention_attention_output_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpDassignvariableop_14_transformer_encoder_25_layer_normalization_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpCassignvariableop_15_transformer_encoder_25_layer_normalization_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpFassignvariableop_16_transformer_encoder_25_layer_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpEassignvariableop_17_transformer_encoder_25_layer_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp7assignvariableop_18_transformer_encoder_25_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp5assignvariableop_19_transformer_encoder_25_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp9assignvariableop_20_transformer_encoder_25_dense_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp7assignvariableop_21_transformer_encoder_25_dense_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpLassignvariableop_22_transformer_decoder_25_multi_head_attention_query_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpJassignvariableop_23_transformer_decoder_25_multi_head_attention_query_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpJassignvariableop_24_transformer_decoder_25_multi_head_attention_key_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpHassignvariableop_25_transformer_decoder_25_multi_head_attention_key_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpLassignvariableop_26_transformer_decoder_25_multi_head_attention_value_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpJassignvariableop_27_transformer_decoder_25_multi_head_attention_value_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpWassignvariableop_28_transformer_decoder_25_multi_head_attention_attention_output_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpUassignvariableop_29_transformer_decoder_25_multi_head_attention_attention_output_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpDassignvariableop_30_transformer_decoder_25_layer_normalization_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpCassignvariableop_31_transformer_decoder_25_layer_normalization_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpFassignvariableop_32_transformer_decoder_25_layer_normalization_1_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpEassignvariableop_33_transformer_decoder_25_layer_normalization_1_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_transformer_decoder_25_dense_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_transformer_decoder_25_dense_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp9assignvariableop_36_transformer_decoder_25_dense_1_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_transformer_decoder_25_dense_1_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_adam_iterIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_beta_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_decayIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp&assignvariableop_42_adam_learning_rateIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
StatefulPartitionedCallStatefulPartitionedCallmutablehashtableRestoreV2:tensors:43RestoreV2:tensors:44"/device:CPU:0*
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
'__inference_restore_from_tensors_505261_
Identity_43IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_totalIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_countIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_14_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_14_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpWassignvariableop_49_adam_token_and_position_embedding_23_token_embedding24_embeddings_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpZassignvariableop_50_adam_token_and_position_embedding_23_position_embedding24_embeddings_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpWassignvariableop_51_adam_token_and_position_embedding_24_token_embedding25_embeddings_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpZassignvariableop_52_adam_token_and_position_embedding_24_position_embedding25_embeddings_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpSassignvariableop_53_adam_transformer_encoder_25_multi_head_attention_query_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpQassignvariableop_54_adam_transformer_encoder_25_multi_head_attention_query_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpQassignvariableop_55_adam_transformer_encoder_25_multi_head_attention_key_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpOassignvariableop_56_adam_transformer_encoder_25_multi_head_attention_key_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpSassignvariableop_57_adam_transformer_encoder_25_multi_head_attention_value_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpQassignvariableop_58_adam_transformer_encoder_25_multi_head_attention_value_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp^assignvariableop_59_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp\assignvariableop_60_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpKassignvariableop_61_adam_transformer_encoder_25_layer_normalization_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpJassignvariableop_62_adam_transformer_encoder_25_layer_normalization_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOpMassignvariableop_63_adam_transformer_encoder_25_layer_normalization_1_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpLassignvariableop_64_adam_transformer_encoder_25_layer_normalization_1_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp>assignvariableop_65_adam_transformer_encoder_25_dense_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp<assignvariableop_66_adam_transformer_encoder_25_dense_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp@assignvariableop_67_adam_transformer_encoder_25_dense_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp>assignvariableop_68_adam_transformer_encoder_25_dense_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOpSassignvariableop_69_adam_transformer_decoder_25_multi_head_attention_query_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpQassignvariableop_70_adam_transformer_decoder_25_multi_head_attention_query_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpQassignvariableop_71_adam_transformer_decoder_25_multi_head_attention_key_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpOassignvariableop_72_adam_transformer_decoder_25_multi_head_attention_key_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpSassignvariableop_73_adam_transformer_decoder_25_multi_head_attention_value_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpQassignvariableop_74_adam_transformer_decoder_25_multi_head_attention_value_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp^assignvariableop_75_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp\assignvariableop_76_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOpKassignvariableop_77_adam_transformer_decoder_25_layer_normalization_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOpJassignvariableop_78_adam_transformer_decoder_25_layer_normalization_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOpMassignvariableop_79_adam_transformer_decoder_25_layer_normalization_1_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOpLassignvariableop_80_adam_transformer_decoder_25_layer_normalization_1_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp>assignvariableop_81_adam_transformer_decoder_25_dense_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp<assignvariableop_82_adam_transformer_decoder_25_dense_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp@assignvariableop_83_adam_transformer_decoder_25_dense_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_transformer_decoder_25_dense_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_14_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_14_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOpWassignvariableop_87_adam_token_and_position_embedding_23_token_embedding24_embeddings_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOpZassignvariableop_88_adam_token_and_position_embedding_23_position_embedding24_embeddings_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOpWassignvariableop_89_adam_token_and_position_embedding_24_token_embedding25_embeddings_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOpZassignvariableop_90_adam_token_and_position_embedding_24_position_embedding25_embeddings_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOpSassignvariableop_91_adam_transformer_encoder_25_multi_head_attention_query_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOpQassignvariableop_92_adam_transformer_encoder_25_multi_head_attention_query_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOpQassignvariableop_93_adam_transformer_encoder_25_multi_head_attention_key_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOpOassignvariableop_94_adam_transformer_encoder_25_multi_head_attention_key_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOpSassignvariableop_95_adam_transformer_encoder_25_multi_head_attention_value_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOpQassignvariableop_96_adam_transformer_encoder_25_multi_head_attention_value_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp^assignvariableop_97_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0`
Identity_98IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp\assignvariableop_98_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0`
Identity_99IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOpKassignvariableop_99_adam_transformer_encoder_25_layer_normalization_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOpKassignvariableop_100_adam_transformer_encoder_25_layer_normalization_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOpNassignvariableop_101_adam_transformer_encoder_25_layer_normalization_1_gamma_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOpMassignvariableop_102_adam_transformer_encoder_25_layer_normalization_1_beta_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp?assignvariableop_103_adam_transformer_encoder_25_dense_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp=assignvariableop_104_adam_transformer_encoder_25_dense_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOpAassignvariableop_105_adam_transformer_encoder_25_dense_1_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp?assignvariableop_106_adam_transformer_encoder_25_dense_1_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOpTassignvariableop_107_adam_transformer_decoder_25_multi_head_attention_query_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOpRassignvariableop_108_adam_transformer_decoder_25_multi_head_attention_query_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOpRassignvariableop_109_adam_transformer_decoder_25_multi_head_attention_key_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOpPassignvariableop_110_adam_transformer_decoder_25_multi_head_attention_key_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOpTassignvariableop_111_adam_transformer_decoder_25_multi_head_attention_value_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOpRassignvariableop_112_adam_transformer_decoder_25_multi_head_attention_value_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp_assignvariableop_113_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp]assignvariableop_114_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOpLassignvariableop_115_adam_transformer_decoder_25_layer_normalization_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOpKassignvariableop_116_adam_transformer_decoder_25_layer_normalization_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOpNassignvariableop_117_adam_transformer_decoder_25_layer_normalization_1_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOpMassignvariableop_118_adam_transformer_decoder_25_layer_normalization_1_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp?assignvariableop_119_adam_transformer_decoder_25_dense_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp=assignvariableop_120_adam_transformer_decoder_25_dense_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOpAassignvariableop_121_adam_transformer_decoder_25_dense_1_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp?assignvariableop_122_adam_transformer_decoder_25_dense_1_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_123Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp^StatefulPartitionedCall"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_124IdentityIdentity_123:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "%
identity_124Identity_124:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_122AssignVariableOp_1222*
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
??
?R
__inference__traced_save_505035
file_prefix.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop[
Wsavev2_token_and_position_embedding_23_token_embedding24_embeddings_read_readvariableop^
Zsavev2_token_and_position_embedding_23_position_embedding24_embeddings_read_readvariableop[
Wsavev2_token_and_position_embedding_24_token_embedding25_embeddings_read_readvariableop^
Zsavev2_token_and_position_embedding_24_position_embedding25_embeddings_read_readvariableopW
Ssavev2_transformer_encoder_25_multi_head_attention_query_kernel_read_readvariableopU
Qsavev2_transformer_encoder_25_multi_head_attention_query_bias_read_readvariableopU
Qsavev2_transformer_encoder_25_multi_head_attention_key_kernel_read_readvariableopS
Osavev2_transformer_encoder_25_multi_head_attention_key_bias_read_readvariableopW
Ssavev2_transformer_encoder_25_multi_head_attention_value_kernel_read_readvariableopU
Qsavev2_transformer_encoder_25_multi_head_attention_value_bias_read_readvariableopb
^savev2_transformer_encoder_25_multi_head_attention_attention_output_kernel_read_readvariableop`
\savev2_transformer_encoder_25_multi_head_attention_attention_output_bias_read_readvariableopO
Ksavev2_transformer_encoder_25_layer_normalization_gamma_read_readvariableopN
Jsavev2_transformer_encoder_25_layer_normalization_beta_read_readvariableopQ
Msavev2_transformer_encoder_25_layer_normalization_1_gamma_read_readvariableopP
Lsavev2_transformer_encoder_25_layer_normalization_1_beta_read_readvariableopB
>savev2_transformer_encoder_25_dense_kernel_read_readvariableop@
<savev2_transformer_encoder_25_dense_bias_read_readvariableopD
@savev2_transformer_encoder_25_dense_1_kernel_read_readvariableopB
>savev2_transformer_encoder_25_dense_1_bias_read_readvariableopW
Ssavev2_transformer_decoder_25_multi_head_attention_query_kernel_read_readvariableopU
Qsavev2_transformer_decoder_25_multi_head_attention_query_bias_read_readvariableopU
Qsavev2_transformer_decoder_25_multi_head_attention_key_kernel_read_readvariableopS
Osavev2_transformer_decoder_25_multi_head_attention_key_bias_read_readvariableopW
Ssavev2_transformer_decoder_25_multi_head_attention_value_kernel_read_readvariableopU
Qsavev2_transformer_decoder_25_multi_head_attention_value_bias_read_readvariableopb
^savev2_transformer_decoder_25_multi_head_attention_attention_output_kernel_read_readvariableop`
\savev2_transformer_decoder_25_multi_head_attention_attention_output_bias_read_readvariableopO
Ksavev2_transformer_decoder_25_layer_normalization_gamma_read_readvariableopN
Jsavev2_transformer_decoder_25_layer_normalization_beta_read_readvariableopQ
Msavev2_transformer_decoder_25_layer_normalization_1_gamma_read_readvariableopP
Lsavev2_transformer_decoder_25_layer_normalization_1_beta_read_readvariableopB
>savev2_transformer_decoder_25_dense_kernel_read_readvariableop@
<savev2_transformer_decoder_25_dense_bias_read_readvariableopD
@savev2_transformer_decoder_25_dense_1_kernel_read_readvariableopB
>savev2_transformer_decoder_25_dense_1_bias_read_readvariableop(
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
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableopb
^savev2_adam_token_and_position_embedding_23_token_embedding24_embeddings_m_read_readvariableope
asavev2_adam_token_and_position_embedding_23_position_embedding24_embeddings_m_read_readvariableopb
^savev2_adam_token_and_position_embedding_24_token_embedding25_embeddings_m_read_readvariableope
asavev2_adam_token_and_position_embedding_24_position_embedding25_embeddings_m_read_readvariableop^
Zsavev2_adam_transformer_encoder_25_multi_head_attention_query_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_encoder_25_multi_head_attention_query_bias_m_read_readvariableop\
Xsavev2_adam_transformer_encoder_25_multi_head_attention_key_kernel_m_read_readvariableopZ
Vsavev2_adam_transformer_encoder_25_multi_head_attention_key_bias_m_read_readvariableop^
Zsavev2_adam_transformer_encoder_25_multi_head_attention_value_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_encoder_25_multi_head_attention_value_bias_m_read_readvariableopi
esavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_m_read_readvariableopg
csavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_m_read_readvariableopV
Rsavev2_adam_transformer_encoder_25_layer_normalization_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_encoder_25_layer_normalization_beta_m_read_readvariableopX
Tsavev2_adam_transformer_encoder_25_layer_normalization_1_gamma_m_read_readvariableopW
Ssavev2_adam_transformer_encoder_25_layer_normalization_1_beta_m_read_readvariableopI
Esavev2_adam_transformer_encoder_25_dense_kernel_m_read_readvariableopG
Csavev2_adam_transformer_encoder_25_dense_bias_m_read_readvariableopK
Gsavev2_adam_transformer_encoder_25_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_transformer_encoder_25_dense_1_bias_m_read_readvariableop^
Zsavev2_adam_transformer_decoder_25_multi_head_attention_query_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_decoder_25_multi_head_attention_query_bias_m_read_readvariableop\
Xsavev2_adam_transformer_decoder_25_multi_head_attention_key_kernel_m_read_readvariableopZ
Vsavev2_adam_transformer_decoder_25_multi_head_attention_key_bias_m_read_readvariableop^
Zsavev2_adam_transformer_decoder_25_multi_head_attention_value_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_decoder_25_multi_head_attention_value_bias_m_read_readvariableopi
esavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_m_read_readvariableopg
csavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_m_read_readvariableopV
Rsavev2_adam_transformer_decoder_25_layer_normalization_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_25_layer_normalization_beta_m_read_readvariableopX
Tsavev2_adam_transformer_decoder_25_layer_normalization_1_gamma_m_read_readvariableopW
Ssavev2_adam_transformer_decoder_25_layer_normalization_1_beta_m_read_readvariableopI
Esavev2_adam_transformer_decoder_25_dense_kernel_m_read_readvariableopG
Csavev2_adam_transformer_decoder_25_dense_bias_m_read_readvariableopK
Gsavev2_adam_transformer_decoder_25_dense_1_kernel_m_read_readvariableopI
Esavev2_adam_transformer_decoder_25_dense_1_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableopb
^savev2_adam_token_and_position_embedding_23_token_embedding24_embeddings_v_read_readvariableope
asavev2_adam_token_and_position_embedding_23_position_embedding24_embeddings_v_read_readvariableopb
^savev2_adam_token_and_position_embedding_24_token_embedding25_embeddings_v_read_readvariableope
asavev2_adam_token_and_position_embedding_24_position_embedding25_embeddings_v_read_readvariableop^
Zsavev2_adam_transformer_encoder_25_multi_head_attention_query_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_encoder_25_multi_head_attention_query_bias_v_read_readvariableop\
Xsavev2_adam_transformer_encoder_25_multi_head_attention_key_kernel_v_read_readvariableopZ
Vsavev2_adam_transformer_encoder_25_multi_head_attention_key_bias_v_read_readvariableop^
Zsavev2_adam_transformer_encoder_25_multi_head_attention_value_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_encoder_25_multi_head_attention_value_bias_v_read_readvariableopi
esavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_v_read_readvariableopg
csavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_v_read_readvariableopV
Rsavev2_adam_transformer_encoder_25_layer_normalization_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_encoder_25_layer_normalization_beta_v_read_readvariableopX
Tsavev2_adam_transformer_encoder_25_layer_normalization_1_gamma_v_read_readvariableopW
Ssavev2_adam_transformer_encoder_25_layer_normalization_1_beta_v_read_readvariableopI
Esavev2_adam_transformer_encoder_25_dense_kernel_v_read_readvariableopG
Csavev2_adam_transformer_encoder_25_dense_bias_v_read_readvariableopK
Gsavev2_adam_transformer_encoder_25_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_transformer_encoder_25_dense_1_bias_v_read_readvariableop^
Zsavev2_adam_transformer_decoder_25_multi_head_attention_query_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_decoder_25_multi_head_attention_query_bias_v_read_readvariableop\
Xsavev2_adam_transformer_decoder_25_multi_head_attention_key_kernel_v_read_readvariableopZ
Vsavev2_adam_transformer_decoder_25_multi_head_attention_key_bias_v_read_readvariableop^
Zsavev2_adam_transformer_decoder_25_multi_head_attention_value_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_decoder_25_multi_head_attention_value_bias_v_read_readvariableopi
esavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_v_read_readvariableopg
csavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_v_read_readvariableopV
Rsavev2_adam_transformer_decoder_25_layer_normalization_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_25_layer_normalization_beta_v_read_readvariableopX
Tsavev2_adam_transformer_decoder_25_layer_normalization_1_gamma_v_read_readvariableopW
Ssavev2_adam_transformer_decoder_25_layer_normalization_1_beta_v_read_readvariableopI
Esavev2_adam_transformer_decoder_25_dense_kernel_v_read_readvariableopG
Csavev2_adam_transformer_decoder_25_dense_bias_v_read_readvariableopK
Gsavev2_adam_transformer_decoder_25_dense_1_kernel_v_read_readvariableopI
Esavev2_adam_transformer_decoder_25_dense_1_bias_v_read_readvariableop
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
: ?;
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?:
value?:B?:~B6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?
value?B?~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?O
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableopWsavev2_token_and_position_embedding_23_token_embedding24_embeddings_read_readvariableopZsavev2_token_and_position_embedding_23_position_embedding24_embeddings_read_readvariableopWsavev2_token_and_position_embedding_24_token_embedding25_embeddings_read_readvariableopZsavev2_token_and_position_embedding_24_position_embedding25_embeddings_read_readvariableopSsavev2_transformer_encoder_25_multi_head_attention_query_kernel_read_readvariableopQsavev2_transformer_encoder_25_multi_head_attention_query_bias_read_readvariableopQsavev2_transformer_encoder_25_multi_head_attention_key_kernel_read_readvariableopOsavev2_transformer_encoder_25_multi_head_attention_key_bias_read_readvariableopSsavev2_transformer_encoder_25_multi_head_attention_value_kernel_read_readvariableopQsavev2_transformer_encoder_25_multi_head_attention_value_bias_read_readvariableop^savev2_transformer_encoder_25_multi_head_attention_attention_output_kernel_read_readvariableop\savev2_transformer_encoder_25_multi_head_attention_attention_output_bias_read_readvariableopKsavev2_transformer_encoder_25_layer_normalization_gamma_read_readvariableopJsavev2_transformer_encoder_25_layer_normalization_beta_read_readvariableopMsavev2_transformer_encoder_25_layer_normalization_1_gamma_read_readvariableopLsavev2_transformer_encoder_25_layer_normalization_1_beta_read_readvariableop>savev2_transformer_encoder_25_dense_kernel_read_readvariableop<savev2_transformer_encoder_25_dense_bias_read_readvariableop@savev2_transformer_encoder_25_dense_1_kernel_read_readvariableop>savev2_transformer_encoder_25_dense_1_bias_read_readvariableopSsavev2_transformer_decoder_25_multi_head_attention_query_kernel_read_readvariableopQsavev2_transformer_decoder_25_multi_head_attention_query_bias_read_readvariableopQsavev2_transformer_decoder_25_multi_head_attention_key_kernel_read_readvariableopOsavev2_transformer_decoder_25_multi_head_attention_key_bias_read_readvariableopSsavev2_transformer_decoder_25_multi_head_attention_value_kernel_read_readvariableopQsavev2_transformer_decoder_25_multi_head_attention_value_bias_read_readvariableop^savev2_transformer_decoder_25_multi_head_attention_attention_output_kernel_read_readvariableop\savev2_transformer_decoder_25_multi_head_attention_attention_output_bias_read_readvariableopKsavev2_transformer_decoder_25_layer_normalization_gamma_read_readvariableopJsavev2_transformer_decoder_25_layer_normalization_beta_read_readvariableopMsavev2_transformer_decoder_25_layer_normalization_1_gamma_read_readvariableopLsavev2_transformer_decoder_25_layer_normalization_1_beta_read_readvariableop>savev2_transformer_decoder_25_dense_kernel_read_readvariableop<savev2_transformer_decoder_25_dense_bias_read_readvariableop@savev2_transformer_decoder_25_dense_1_kernel_read_readvariableop>savev2_transformer_decoder_25_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop^savev2_adam_token_and_position_embedding_23_token_embedding24_embeddings_m_read_readvariableopasavev2_adam_token_and_position_embedding_23_position_embedding24_embeddings_m_read_readvariableop^savev2_adam_token_and_position_embedding_24_token_embedding25_embeddings_m_read_readvariableopasavev2_adam_token_and_position_embedding_24_position_embedding25_embeddings_m_read_readvariableopZsavev2_adam_transformer_encoder_25_multi_head_attention_query_kernel_m_read_readvariableopXsavev2_adam_transformer_encoder_25_multi_head_attention_query_bias_m_read_readvariableopXsavev2_adam_transformer_encoder_25_multi_head_attention_key_kernel_m_read_readvariableopVsavev2_adam_transformer_encoder_25_multi_head_attention_key_bias_m_read_readvariableopZsavev2_adam_transformer_encoder_25_multi_head_attention_value_kernel_m_read_readvariableopXsavev2_adam_transformer_encoder_25_multi_head_attention_value_bias_m_read_readvariableopesavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_m_read_readvariableopcsavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_m_read_readvariableopRsavev2_adam_transformer_encoder_25_layer_normalization_gamma_m_read_readvariableopQsavev2_adam_transformer_encoder_25_layer_normalization_beta_m_read_readvariableopTsavev2_adam_transformer_encoder_25_layer_normalization_1_gamma_m_read_readvariableopSsavev2_adam_transformer_encoder_25_layer_normalization_1_beta_m_read_readvariableopEsavev2_adam_transformer_encoder_25_dense_kernel_m_read_readvariableopCsavev2_adam_transformer_encoder_25_dense_bias_m_read_readvariableopGsavev2_adam_transformer_encoder_25_dense_1_kernel_m_read_readvariableopEsavev2_adam_transformer_encoder_25_dense_1_bias_m_read_readvariableopZsavev2_adam_transformer_decoder_25_multi_head_attention_query_kernel_m_read_readvariableopXsavev2_adam_transformer_decoder_25_multi_head_attention_query_bias_m_read_readvariableopXsavev2_adam_transformer_decoder_25_multi_head_attention_key_kernel_m_read_readvariableopVsavev2_adam_transformer_decoder_25_multi_head_attention_key_bias_m_read_readvariableopZsavev2_adam_transformer_decoder_25_multi_head_attention_value_kernel_m_read_readvariableopXsavev2_adam_transformer_decoder_25_multi_head_attention_value_bias_m_read_readvariableopesavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_m_read_readvariableopcsavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_m_read_readvariableopRsavev2_adam_transformer_decoder_25_layer_normalization_gamma_m_read_readvariableopQsavev2_adam_transformer_decoder_25_layer_normalization_beta_m_read_readvariableopTsavev2_adam_transformer_decoder_25_layer_normalization_1_gamma_m_read_readvariableopSsavev2_adam_transformer_decoder_25_layer_normalization_1_beta_m_read_readvariableopEsavev2_adam_transformer_decoder_25_dense_kernel_m_read_readvariableopCsavev2_adam_transformer_decoder_25_dense_bias_m_read_readvariableopGsavev2_adam_transformer_decoder_25_dense_1_kernel_m_read_readvariableopEsavev2_adam_transformer_decoder_25_dense_1_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop^savev2_adam_token_and_position_embedding_23_token_embedding24_embeddings_v_read_readvariableopasavev2_adam_token_and_position_embedding_23_position_embedding24_embeddings_v_read_readvariableop^savev2_adam_token_and_position_embedding_24_token_embedding25_embeddings_v_read_readvariableopasavev2_adam_token_and_position_embedding_24_position_embedding25_embeddings_v_read_readvariableopZsavev2_adam_transformer_encoder_25_multi_head_attention_query_kernel_v_read_readvariableopXsavev2_adam_transformer_encoder_25_multi_head_attention_query_bias_v_read_readvariableopXsavev2_adam_transformer_encoder_25_multi_head_attention_key_kernel_v_read_readvariableopVsavev2_adam_transformer_encoder_25_multi_head_attention_key_bias_v_read_readvariableopZsavev2_adam_transformer_encoder_25_multi_head_attention_value_kernel_v_read_readvariableopXsavev2_adam_transformer_encoder_25_multi_head_attention_value_bias_v_read_readvariableopesavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_kernel_v_read_readvariableopcsavev2_adam_transformer_encoder_25_multi_head_attention_attention_output_bias_v_read_readvariableopRsavev2_adam_transformer_encoder_25_layer_normalization_gamma_v_read_readvariableopQsavev2_adam_transformer_encoder_25_layer_normalization_beta_v_read_readvariableopTsavev2_adam_transformer_encoder_25_layer_normalization_1_gamma_v_read_readvariableopSsavev2_adam_transformer_encoder_25_layer_normalization_1_beta_v_read_readvariableopEsavev2_adam_transformer_encoder_25_dense_kernel_v_read_readvariableopCsavev2_adam_transformer_encoder_25_dense_bias_v_read_readvariableopGsavev2_adam_transformer_encoder_25_dense_1_kernel_v_read_readvariableopEsavev2_adam_transformer_encoder_25_dense_1_bias_v_read_readvariableopZsavev2_adam_transformer_decoder_25_multi_head_attention_query_kernel_v_read_readvariableopXsavev2_adam_transformer_decoder_25_multi_head_attention_query_bias_v_read_readvariableopXsavev2_adam_transformer_decoder_25_multi_head_attention_key_kernel_v_read_readvariableopVsavev2_adam_transformer_decoder_25_multi_head_attention_key_bias_v_read_readvariableopZsavev2_adam_transformer_decoder_25_multi_head_attention_value_kernel_v_read_readvariableopXsavev2_adam_transformer_decoder_25_multi_head_attention_value_bias_v_read_readvariableopesavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_kernel_v_read_readvariableopcsavev2_adam_transformer_decoder_25_multi_head_attention_attention_output_bias_v_read_readvariableopRsavev2_adam_transformer_decoder_25_layer_normalization_gamma_v_read_readvariableopQsavev2_adam_transformer_decoder_25_layer_normalization_beta_v_read_readvariableopTsavev2_adam_transformer_decoder_25_layer_normalization_1_gamma_v_read_readvariableopSsavev2_adam_transformer_decoder_25_layer_normalization_1_beta_v_read_readvariableopEsavev2_adam_transformer_decoder_25_dense_kernel_v_read_readvariableopCsavev2_adam_transformer_decoder_25_dense_bias_v_read_readvariableopGsavev2_adam_transformer_decoder_25_dense_1_kernel_v_read_readvariableopEsavev2_adam_transformer_decoder_25_dense_1_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2~		?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :  : :	? :  : :  : :: :: :: : : : : : : @:@:@ : : :: :: :: : : : : : : @:@:@ : : : : : : ::: : : : :  : :	? :  : :  : :: :: :: : : : : : : @:@:@ : : :: :: :: : : : : : : @:@:@ : :  : :	? :  : :  : :: :: :: : : : : : : @:@:@ : : :: :: :: : : : : : : @:@:@ : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :%!

_output_shapes
:	? :$ 

_output_shapes

:  :$ 

_output_shapes

: :$ 

_output_shapes

:  :($
"
_output_shapes
: :$ 

_output_shapes

::(	$
"
_output_shapes
: :$
 

_output_shapes

::($
"
_output_shapes
: :$ 

_output_shapes

::($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :($
"
_output_shapes
: :$ 

_output_shapes

::($
"
_output_shapes
: :$ 

_output_shapes

::($
"
_output_shapes
: :$ 

_output_shapes

::($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: :$# 

_output_shapes

: @: $

_output_shapes
:@:$% 

_output_shapes

:@ : &

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
::-

_output_shapes
::.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :$2 

_output_shapes

:  : 3

_output_shapes
: :%4!

_output_shapes
:	? :$5 

_output_shapes

:  :$6 

_output_shapes

: :$7 

_output_shapes

:  :(8$
"
_output_shapes
: :$9 

_output_shapes

::(:$
"
_output_shapes
: :$; 

_output_shapes

::(<$
"
_output_shapes
: :$= 

_output_shapes

::(>$
"
_output_shapes
: : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: : B

_output_shapes
: : C

_output_shapes
: :$D 

_output_shapes

: @: E

_output_shapes
:@:$F 

_output_shapes

:@ : G

_output_shapes
: :(H$
"
_output_shapes
: :$I 

_output_shapes

::(J$
"
_output_shapes
: :$K 

_output_shapes

::(L$
"
_output_shapes
: :$M 

_output_shapes

::(N$
"
_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: : R

_output_shapes
: : S

_output_shapes
: :$T 

_output_shapes

: @: U

_output_shapes
:@:$V 

_output_shapes

:@ : W

_output_shapes
: :$X 

_output_shapes

:  : Y

_output_shapes
: :%Z!

_output_shapes
:	? :$[ 

_output_shapes

:  :$\ 

_output_shapes

: :$] 

_output_shapes

:  :(^$
"
_output_shapes
: :$_ 

_output_shapes

::(`$
"
_output_shapes
: :$a 

_output_shapes

::(b$
"
_output_shapes
: :$c 

_output_shapes

::(d$
"
_output_shapes
: : e

_output_shapes
: : f

_output_shapes
: : g

_output_shapes
: : h

_output_shapes
: : i

_output_shapes
: :$j 

_output_shapes

: @: k

_output_shapes
:@:$l 

_output_shapes

:@ : m

_output_shapes
: :(n$
"
_output_shapes
: :$o 

_output_shapes

::(p$
"
_output_shapes
: :$q 

_output_shapes

::(r$
"
_output_shapes
: :$s 

_output_shapes

::(t$
"
_output_shapes
: : u

_output_shapes
: : v

_output_shapes
: : w

_output_shapes
: : x

_output_shapes
: : y

_output_shapes
: :$z 

_output_shapes

: @: {

_output_shapes
:@:$| 

_output_shapes

:@ : }

_output_shapes
: :~

_output_shapes
: 
?
W
;__inference_global_average_pooling1d_9_layer_call_fn_504541

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
GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_500673i
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
?
?
__inference_save_fn_504619
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
?!
?
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_503697

inputs	<
)token_embedding24_embedding_lookup_503673:	? >
,position_embedding24_readvariableop_resource:  
identity??#position_embedding24/ReadVariableOp?"token_embedding24/embedding_lookup?
"token_embedding24/embedding_lookupResourceGather)token_embedding24_embedding_lookup_503673inputs*
Tindices0	*<
_class2
0.loc:@token_embedding24/embedding_lookup/503673*+
_output_shapes
:?????????  *
dtype0?
+token_embedding24/embedding_lookup/IdentityIdentity+token_embedding24/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding24/embedding_lookup/503673*+
_output_shapes
:?????????  ?
-token_embedding24/embedding_lookup/Identity_1Identity4token_embedding24/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
position_embedding24/ShapeShape6token_embedding24/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding24/strided_sliceStridedSlice#position_embedding24/Shape:output:01position_embedding24/strided_slice/stack:output:03position_embedding24/strided_slice/stack_1:output:03position_embedding24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding24/ReadVariableOpReadVariableOp,position_embedding24_readvariableop_resource*
_output_shapes

:  *
dtype0\
position_embedding24/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding24/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding24/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding24/strided_slice_1/stackPack#position_embedding24/Const:output:05position_embedding24/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding24/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding24/strided_slice_1/stack_1Pack+position_embedding24/strided_slice:output:07position_embedding24/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding24/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding24/strided_slice_1/stack_2Pack%position_embedding24/Const_1:output:07position_embedding24/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding24/strided_slice_1StridedSlice+position_embedding24/ReadVariableOp:value:03position_embedding24/strided_slice_1/stack:output:05position_embedding24/strided_slice_1/stack_1:output:05position_embedding24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
 position_embedding24/BroadcastToBroadcastTo-position_embedding24/strided_slice_1:output:0#position_embedding24/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
addAddV26token_embedding24/embedding_lookup/Identity_1:output:0)position_embedding24/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp$^position_embedding24/ReadVariableOp#^token_embedding24/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding24/ReadVariableOp#position_embedding24/ReadVariableOp2H
"token_embedding24/embedding_lookup"token_embedding24/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
;
__inference__creator_504572
identity??
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name393026*
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
?!
?
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_500760

inputs	<
)token_embedding24_embedding_lookup_500736:	? >
,position_embedding24_readvariableop_resource:  
identity??#position_embedding24/ReadVariableOp?"token_embedding24/embedding_lookup?
"token_embedding24/embedding_lookupResourceGather)token_embedding24_embedding_lookup_500736inputs*
Tindices0	*<
_class2
0.loc:@token_embedding24/embedding_lookup/500736*+
_output_shapes
:?????????  *
dtype0?
+token_embedding24/embedding_lookup/IdentityIdentity+token_embedding24/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding24/embedding_lookup/500736*+
_output_shapes
:?????????  ?
-token_embedding24/embedding_lookup/Identity_1Identity4token_embedding24/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
position_embedding24/ShapeShape6token_embedding24/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding24/strided_sliceStridedSlice#position_embedding24/Shape:output:01position_embedding24/strided_slice/stack:output:03position_embedding24/strided_slice/stack_1:output:03position_embedding24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding24/ReadVariableOpReadVariableOp,position_embedding24_readvariableop_resource*
_output_shapes

:  *
dtype0\
position_embedding24/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding24/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding24/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding24/strided_slice_1/stackPack#position_embedding24/Const:output:05position_embedding24/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding24/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding24/strided_slice_1/stack_1Pack+position_embedding24/strided_slice:output:07position_embedding24/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding24/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding24/strided_slice_1/stack_2Pack%position_embedding24/Const_1:output:07position_embedding24/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding24/strided_slice_1StridedSlice+position_embedding24/ReadVariableOp:value:03position_embedding24/strided_slice_1/stack:output:05position_embedding24/strided_slice_1/stack_1:output:05position_embedding24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
 position_embedding24/BroadcastToBroadcastTo-position_embedding24/strided_slice_1:output:0#position_embedding24/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
addAddV26token_embedding24/embedding_lookup/Identity_1:output:0)position_embedding24/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp$^position_embedding24/ReadVariableOp#^token_embedding24/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding24/ReadVariableOp#position_embedding24/ReadVariableOp2H
"token_embedding24/embedding_lookup"token_embedding24/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?	
(__inference_model_9_layer_call_fn_502725
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	? 
	unknown_4:  
	unknown_5: 
	unknown_6:  
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22:  

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39:  

unknown_40: 
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
unknown_40*7
Tin0
.2,		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *H
_read_only_resource_inputs*
(&	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_9_layer_call_and_return_conditional_losses_501195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
?
?	
$__inference_signature_wrapper_502587

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	? 
	unknown_4:  
	unknown_5: 
	unknown_6:  
	unknown_7: 
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@ 

unknown_20: 

unknown_21: 

unknown_22:  

unknown_23: 

unknown_24: 

unknown_25: 

unknown_26: 

unknown_27: 

unknown_28: 

unknown_29: 

unknown_30: 

unknown_31: 

unknown_32: 

unknown_33: @

unknown_34:@

unknown_35:@ 

unknown_36: 

unknown_37: 

unknown_38: 

unknown_39:  

unknown_40: 
identity??StatefulPartitionedCall?
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
unknown_40*7
Tin0
.2,		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *H
_read_only_resource_inputs*
(&	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_500663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
?

?
'__inference_restore_from_tensors_505261M
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
??
?
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501526
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
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
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  o
addAddV2dropout/dropout/Mul_1:z:0decoder_sequence*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
*
_user_specified_namedecoder_sequence
?
r
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_500673

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
D__inference_dense_14_layer_call_and_return_conditional_losses_504567

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:????????? `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_transformer_decoder_25_layer_call_fn_504132
decoder_sequence
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: @

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 
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
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501142s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:?????????  
*
_user_specified_namedecoder_sequence
??
?
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504342
decoder_sequenceV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
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
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsumdecoder_sequence=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsumdecoder_sequence?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
dropout/IdentityIdentity-multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:?????????  o
addAddV2dropout/Identity:output:0decoder_sequence*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  n
dropout_1/IdentityIdentitydense_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/Identity:output:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
*
_user_specified_namedecoder_sequence
?
r
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_504547

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
??
?
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_504095

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  e
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
 
_user_specified_nameinputs
?
-
__inference__destroyer_504600
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
/
__inference__initializer_504595
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
?"
?
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_500794

inputs;
)token_embedding25_embedding_lookup_500770: >
,position_embedding25_readvariableop_resource:  
identity??#position_embedding25/ReadVariableOp?"token_embedding25/embedding_lookupg
token_embedding25/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
"token_embedding25/embedding_lookupResourceGather)token_embedding25_embedding_lookup_500770token_embedding25/Cast:y:0*
Tindices0*<
_class2
0.loc:@token_embedding25/embedding_lookup/500770*+
_output_shapes
:?????????  *
dtype0?
+token_embedding25/embedding_lookup/IdentityIdentity+token_embedding25/embedding_lookup:output:0*
T0*<
_class2
0.loc:@token_embedding25/embedding_lookup/500770*+
_output_shapes
:?????????  ?
-token_embedding25/embedding_lookup/Identity_1Identity4token_embedding25/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????  ?
position_embedding25/ShapeShape6token_embedding25/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:{
(position_embedding25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
*position_embedding25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????t
*position_embedding25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"position_embedding25/strided_sliceStridedSlice#position_embedding25/Shape:output:01position_embedding25/strided_slice/stack:output:03position_embedding25/strided_slice/stack_1:output:03position_embedding25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
#position_embedding25/ReadVariableOpReadVariableOp,position_embedding25_readvariableop_resource*
_output_shapes

:  *
dtype0\
position_embedding25/ConstConst*
_output_shapes
: *
dtype0*
value	B : ^
position_embedding25/Const_1Const*
_output_shapes
: *
dtype0*
value	B :n
,position_embedding25/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
*position_embedding25/strided_slice_1/stackPack#position_embedding25/Const:output:05position_embedding25/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding25/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
,position_embedding25/strided_slice_1/stack_1Pack+position_embedding25/strided_slice:output:07position_embedding25/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:p
.position_embedding25/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
,position_embedding25/strided_slice_1/stack_2Pack%position_embedding25/Const_1:output:07position_embedding25/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
$position_embedding25/strided_slice_1StridedSlice+position_embedding25/ReadVariableOp:value:03position_embedding25/strided_slice_1/stack:output:05position_embedding25/strided_slice_1/stack_1:output:05position_embedding25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:  *

begin_mask*
end_mask?
 position_embedding25/BroadcastToBroadcastTo-position_embedding25/strided_slice_1:output:0#position_embedding25/Shape:output:0*
T0*+
_output_shapes
:?????????  ?
addAddV26token_embedding25/embedding_lookup/Identity_1:output:0)position_embedding25/BroadcastTo:output:0*
T0*+
_output_shapes
:?????????  Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp$^position_embedding25/ReadVariableOp#^token_embedding25/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2J
#position_embedding25/ReadVariableOp#position_embedding25/ReadVariableOp2H
"token_embedding25/embedding_lookup"token_embedding25/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_transformer_encoder_25_layer_call_fn_503783

inputs
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: @

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: 

unknown_14: 
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
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_500935s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????  
 
_user_specified_nameinputs
ܒ
?
C__inference_model_9_layer_call_and_return_conditional_losses_501195

inputs
inputs_1S
Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_9_string_lookup_9_equal_y3
/text_vectorization_9_string_lookup_9_selectv2_t	9
&token_and_position_embedding_23_500761:	? 8
&token_and_position_embedding_23_500763:  8
&token_and_position_embedding_24_500795: 8
&token_and_position_embedding_24_500797:  3
transformer_encoder_25_500936: /
transformer_encoder_25_500938:3
transformer_encoder_25_500940: /
transformer_encoder_25_500942:3
transformer_encoder_25_500944: /
transformer_encoder_25_500946:3
transformer_encoder_25_500948: +
transformer_encoder_25_500950: +
transformer_encoder_25_500952: +
transformer_encoder_25_500954: /
transformer_encoder_25_500956: @+
transformer_encoder_25_500958:@/
transformer_encoder_25_500960:@ +
transformer_encoder_25_500962: +
transformer_encoder_25_500964: +
transformer_encoder_25_500966: 3
transformer_decoder_25_501143: /
transformer_decoder_25_501145:3
transformer_decoder_25_501147: /
transformer_decoder_25_501149:3
transformer_decoder_25_501151: /
transformer_decoder_25_501153:3
transformer_decoder_25_501155: +
transformer_decoder_25_501157: +
transformer_decoder_25_501159: +
transformer_decoder_25_501161: /
transformer_decoder_25_501163: @+
transformer_decoder_25_501165:@/
transformer_decoder_25_501167:@ +
transformer_decoder_25_501169: +
transformer_decoder_25_501171: +
transformer_decoder_25_501173: !
dense_14_501189:  
dense_14_501191: 
identity?? dense_14/StatefulPartitionedCall?Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?7token_and_position_embedding_23/StatefulPartitionedCall?7token_and_position_embedding_24/StatefulPartitionedCall?.transformer_decoder_25/StatefulPartitionedCall?.transformer_encoder_25/StatefulPartitionedCall}
text_vectorization_9/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_9/StringSplit/StringSplitV2StringSplitV2%text_vectorization_9/Squeeze:output:0/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_9/StringSplit/strided_sliceStridedSlice8text_vectorization_9/StringSplit/StringSplitV2:indices:0=text_vectorization_9/StringSplit/strided_slice/stack:output:0?text_vectorization_9/StringSplit/strided_slice/stack_1:output:0?text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_9/StringSplit/strided_slice_1StridedSlice6text_vectorization_9/StringSplit/StringSplitV2:shape:0?text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
itext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
dtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_9/StringSplit/StringSplitV2:values:0Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_9/string_lookup_9/EqualEqual7text_vectorization_9/StringSplit/StringSplitV2:values:0,text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/SelectV2SelectV2.text_vectorization_9/string_lookup_9/Equal:z:0/text_vectorization_9_string_lookup_9_selectv2_tKtext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/IdentityIdentity6text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
8text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_9/RaggedToTensor/Const:output:06text_vectorization_9/string_lookup_9/Identity:output:0:text_vectorization_9/RaggedToTensor/default_value:output:09text_vectorization_9/StringSplit/strided_slice_1:output:07text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_23/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_23_500761&token_and_position_embedding_23_500763*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_500760?
7token_and_position_embedding_24/StatefulPartitionedCallStatefulPartitionedCallinputs_1&token_and_position_embedding_24_500795&token_and_position_embedding_24_500797*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_500794?
add_9/PartitionedCallPartitionedCall@token_and_position_embedding_23/StatefulPartitionedCall:output:0@token_and_position_embedding_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_500806?
.transformer_encoder_25/StatefulPartitionedCallStatefulPartitionedCalladd_9/PartitionedCall:output:0transformer_encoder_25_500936transformer_encoder_25_500938transformer_encoder_25_500940transformer_encoder_25_500942transformer_encoder_25_500944transformer_encoder_25_500946transformer_encoder_25_500948transformer_encoder_25_500950transformer_encoder_25_500952transformer_encoder_25_500954transformer_encoder_25_500956transformer_encoder_25_500958transformer_encoder_25_500960transformer_encoder_25_500962transformer_encoder_25_500964transformer_encoder_25_500966*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_500935?
.transformer_decoder_25/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_25/StatefulPartitionedCall:output:0transformer_decoder_25_501143transformer_decoder_25_501145transformer_decoder_25_501147transformer_decoder_25_501149transformer_decoder_25_501151transformer_decoder_25_501153transformer_decoder_25_501155transformer_decoder_25_501157transformer_decoder_25_501159transformer_decoder_25_501161transformer_decoder_25_501163transformer_decoder_25_501165transformer_decoder_25_501167transformer_decoder_25_501169transformer_decoder_25_501171transformer_decoder_25_501173*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501142?
*global_average_pooling1d_9/PartitionedCallPartitionedCall7transformer_decoder_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_500673?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_9/PartitionedCall:output:0dense_14_501189dense_14_501191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_501188x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_14/StatefulPartitionedCallC^text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV28^token_and_position_embedding_23/StatefulPartitionedCall8^token_and_position_embedding_24/StatefulPartitionedCall/^transformer_decoder_25/StatefulPartitionedCall/^transformer_encoder_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_23/StatefulPartitionedCall7token_and_position_embedding_23/StatefulPartitionedCall2r
7token_and_position_embedding_24/StatefulPartitionedCall7token_and_position_embedding_24/StatefulPartitionedCall2`
.transformer_decoder_25/StatefulPartitionedCall.transformer_decoder_25/StatefulPartitionedCall2`
.transformer_encoder_25/StatefulPartitionedCall.transformer_encoder_25/StatefulPartitionedCall:O K
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
?
?
@__inference_token_and_position_embedding_24_layer_call_fn_503706

inputs
unknown: 
	unknown_0:  
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_500794s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????  `
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
?
C__inference_model_9_layer_call_and_return_conditional_losses_502353

phrase

token_roleS
Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_9_string_lookup_9_equal_y3
/text_vectorization_9_string_lookup_9_selectv2_t	9
&token_and_position_embedding_23_502269:	? 8
&token_and_position_embedding_23_502271:  8
&token_and_position_embedding_24_502274: 8
&token_and_position_embedding_24_502276:  3
transformer_encoder_25_502280: /
transformer_encoder_25_502282:3
transformer_encoder_25_502284: /
transformer_encoder_25_502286:3
transformer_encoder_25_502288: /
transformer_encoder_25_502290:3
transformer_encoder_25_502292: +
transformer_encoder_25_502294: +
transformer_encoder_25_502296: +
transformer_encoder_25_502298: /
transformer_encoder_25_502300: @+
transformer_encoder_25_502302:@/
transformer_encoder_25_502304:@ +
transformer_encoder_25_502306: +
transformer_encoder_25_502308: +
transformer_encoder_25_502310: 3
transformer_decoder_25_502313: /
transformer_decoder_25_502315:3
transformer_decoder_25_502317: /
transformer_decoder_25_502319:3
transformer_decoder_25_502321: /
transformer_decoder_25_502323:3
transformer_decoder_25_502325: +
transformer_decoder_25_502327: +
transformer_decoder_25_502329: +
transformer_decoder_25_502331: /
transformer_decoder_25_502333: @+
transformer_decoder_25_502335:@/
transformer_decoder_25_502337:@ +
transformer_decoder_25_502339: +
transformer_decoder_25_502341: +
transformer_decoder_25_502343: !
dense_14_502347:  
dense_14_502349: 
identity?? dense_14/StatefulPartitionedCall?Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2?7token_and_position_embedding_23/StatefulPartitionedCall?7token_and_position_embedding_24/StatefulPartitionedCall?.transformer_decoder_25/StatefulPartitionedCall?.transformer_encoder_25/StatefulPartitionedCall}
text_vectorization_9/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????g
&text_vectorization_9/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_9/StringSplit/StringSplitV2StringSplitV2%text_vectorization_9/Squeeze:output:0/text_vectorization_9/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_9/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_9/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_9/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_9/StringSplit/strided_sliceStridedSlice8text_vectorization_9/StringSplit/StringSplitV2:indices:0=text_vectorization_9/StringSplit/strided_slice/stack:output:0?text_vectorization_9/StringSplit/strided_slice/stack_1:output:0?text_vectorization_9/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_9/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_9/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_9/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_9/StringSplit/strided_slice_1StridedSlice6text_vectorization_9/StringSplit/StringSplitV2:shape:0?text_vectorization_9/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_9/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_9/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_9/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
itext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
ctext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshape[text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
dtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_9/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_table_handle7text_vectorization_9/StringSplit/StringSplitV2:values:0Ptext_vectorization_9_string_lookup_9_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_9/string_lookup_9/EqualEqual7text_vectorization_9/StringSplit/StringSplitV2:values:0,text_vectorization_9_string_lookup_9_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/SelectV2SelectV2.text_vectorization_9/string_lookup_9/Equal:z:0/text_vectorization_9_string_lookup_9_selectv2_tKtext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_9/string_lookup_9/IdentityIdentity6text_vectorization_9/string_lookup_9/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_9/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_9/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
8text_vectorization_9/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_9/RaggedToTensor/Const:output:06text_vectorization_9/string_lookup_9/Identity:output:0:text_vectorization_9/RaggedToTensor/default_value:output:09text_vectorization_9/StringSplit/strided_slice_1:output:07text_vectorization_9/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
7token_and_position_embedding_23/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_9/RaggedToTensor/RaggedTensorToTensor:result:0&token_and_position_embedding_23_502269&token_and_position_embedding_23_502271*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_500760?
7token_and_position_embedding_24/StatefulPartitionedCallStatefulPartitionedCall
token_role&token_and_position_embedding_24_502274&token_and_position_embedding_24_502276*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *d
f_R]
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_500794?
add_9/PartitionedCallPartitionedCall@token_and_position_embedding_23/StatefulPartitionedCall:output:0@token_and_position_embedding_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_9_layer_call_and_return_conditional_losses_500806?
.transformer_encoder_25/StatefulPartitionedCallStatefulPartitionedCalladd_9/PartitionedCall:output:0transformer_encoder_25_502280transformer_encoder_25_502282transformer_encoder_25_502284transformer_encoder_25_502286transformer_encoder_25_502288transformer_encoder_25_502290transformer_encoder_25_502292transformer_encoder_25_502294transformer_encoder_25_502296transformer_encoder_25_502298transformer_encoder_25_502300transformer_encoder_25_502302transformer_encoder_25_502304transformer_encoder_25_502306transformer_encoder_25_502308transformer_encoder_25_502310*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_500935?
.transformer_decoder_25/StatefulPartitionedCallStatefulPartitionedCall7transformer_encoder_25/StatefulPartitionedCall:output:0transformer_decoder_25_502313transformer_decoder_25_502315transformer_decoder_25_502317transformer_decoder_25_502319transformer_decoder_25_502321transformer_decoder_25_502323transformer_decoder_25_502325transformer_decoder_25_502327transformer_decoder_25_502329transformer_decoder_25_502331transformer_decoder_25_502333transformer_decoder_25_502335transformer_decoder_25_502337transformer_decoder_25_502339transformer_decoder_25_502341transformer_decoder_25_502343*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????  *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_501142?
*global_average_pooling1d_9/PartitionedCallPartitionedCall7transformer_decoder_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_500673?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_9/PartitionedCall:output:0dense_14_502347dense_14_502349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_14_layer_call_and_return_conditional_losses_501188x
IdentityIdentity)dense_14/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp!^dense_14/StatefulPartitionedCallC^text_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV28^token_and_position_embedding_23/StatefulPartitionedCall8^token_and_position_embedding_24/StatefulPartitionedCall/^transformer_decoder_25/StatefulPartitionedCall/^transformer_encoder_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes|
z:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2?
Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV2Btext_vectorization_9/string_lookup_9/None_Lookup/LookupTableFindV22r
7token_and_position_embedding_23/StatefulPartitionedCall7token_and_position_embedding_23/StatefulPartitionedCall2r
7token_and_position_embedding_24/StatefulPartitionedCall7token_and_position_embedding_24/StatefulPartitionedCall2`
.transformer_decoder_25/StatefulPartitionedCall.transformer_decoder_25/StatefulPartitionedCall2`
.transformer_encoder_25/StatefulPartitionedCall.transformer_encoder_25/StatefulPartitionedCall:O K
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
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_501749

inputsV
@multi_head_attention_query_einsum_einsum_readvariableop_resource: H
6multi_head_attention_query_add_readvariableop_resource:T
>multi_head_attention_key_einsum_einsum_readvariableop_resource: F
4multi_head_attention_key_add_readvariableop_resource:V
@multi_head_attention_value_einsum_einsum_readvariableop_resource: H
6multi_head_attention_value_add_readvariableop_resource:a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource: O
Amulti_head_attention_attention_output_add_readvariableop_resource: G
9layer_normalization_batchnorm_mul_readvariableop_resource: C
5layer_normalization_batchnorm_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource: @3
%dense_biasadd_readvariableop_resource:@;
)dense_1_tensordot_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: I
;layer_normalization_1_batchnorm_mul_readvariableop_resource: E
7layer_normalization_1_batchnorm_readvariableop_resource: 
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp?,layer_normalization/batchnorm/ReadVariableOp?0layer_normalization/batchnorm/mul/ReadVariableOp?.layer_normalization_1/batchnorm/ReadVariableOp?2layer_normalization_1/batchnorm/mul/ReadVariableOp?8multi_head_attention/attention_output/add/ReadVariableOp?Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp?+multi_head_attention/key/add/ReadVariableOp?5multi_head_attention/key/einsum/Einsum/ReadVariableOp?-multi_head_attention/query/add/ReadVariableOp?7multi_head_attention/query/einsum/Einsum/ReadVariableOp?-multi_head_attention/value/add/ReadVariableOp?7multi_head_attention/value/einsum/Einsum/ReadVariableOp?
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/query/einsum/EinsumEinsuminputs?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
&multi_head_attention/key/einsum/EinsumEinsuminputs=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
(multi_head_attention/value/einsum/EinsumEinsuminputs?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? _
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
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
:????????? *
equationacbe,aecd->abcd?
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
: *
dtype0?
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:?????????  *
equationabcd,cde->abe?
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype0?
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout/dropout/MulMul-multi_head_attention/attention_output/add:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  r
dropout/dropout/ShapeShape-multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  e
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  |
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
:?????????  ?
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
: *
dtype0?
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_1Muladd:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  ?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

: @*
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
:?????????  ?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
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
:????????? @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? @`

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? @?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:@ *
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
:????????? @?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? c
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: a
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
:?????????  ?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_1/dropout/MulMuldense_1/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????  _
dropout_1/dropout/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????  *
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
:?????????  ?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????  ?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????  ?
add_1AddV2'layer_normalization/batchnorm/add_1:z:0dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????  ~
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
:?????????  ?
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
: *
dtype0?
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????  ?
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0?
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????  ?
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????  |
IdentityIdentity)layer_normalization_1/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????  ?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????  : : : : : : : : : : : : : : : : 2<
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
:?????????  
 
_user_specified_nameinputs
?
k
A__inference_add_9_layer_call_and_return_conditional_losses_500806

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:?????????  S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????  :?????????  :S O
+
_output_shapes
:?????????  
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????  
 
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
dense_140
StatefulPartitionedCall:0????????? tensorflow/serving/predict:??
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%token_embedding
&position_embedding"
_tf_keras_layer
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3_multi_head_attention_layer
4_attention_layernorm
5_feedforward_layernorm
6_attention_dropout
7_intermediate_dense
8_output_dense
9_output_dropout"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@_self_attention_layer
 A_decoder_attention_layernorm
B_feedforward_layernorm
C_self_attention_dropout
D_intermediate_dense
E_output_dense
F_output_dropout"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
Tbias"
_tf_keras_layer
?
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13
b14
c15
d16
e17
f18
g19
h20
i21
j22
k23
l24
m25
n26
o27
p28
q29
r30
s31
t32
u33
v34
w35
x36
S37
T38"
trackable_list_wrapper
?
U0
V1
W2
X3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
g18
h19
i20
j21
k22
l23
m24
n25
o26
p27
q28
r29
s30
t31
u32
v33
w34
x35
S36
T37"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
~trace_0
trace_1
?trace_2
?trace_32?
(__inference_model_9_layer_call_fn_501282
(__inference_model_9_layer_call_fn_502725
(__inference_model_9_layer_call_fn_502815
(__inference_model_9_layer_call_fn_502217?
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
 z~trace_0ztrace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
C__inference_model_9_layer_call_and_return_conditional_losses_503217
C__inference_model_9_layer_call_and_return_conditional_losses_503661
C__inference_model_9_layer_call_and_return_conditional_losses_502353
C__inference_model_9_layer_call_and_return_conditional_losses_502489?
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
!__inference__wrapped_model_500663phrase
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
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateSm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?"
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
__inference_adapt_step_502635?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
@__inference_token_and_position_embedding_23_layer_call_fn_503670?
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
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_503697?
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
U
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
V
embeddings
Vposition_embeddings"
_tf_keras_layer
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
@__inference_token_and_position_embedding_24_layer_call_fn_503706?
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
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_503734?
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
W
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
X
embeddings
Xposition_embeddings"
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
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
&__inference_add_9_layer_call_fn_503740?
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
A__inference_add_9_layer_call_and_return_conditional_losses_503746?
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
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15"
trackable_list_wrapper
?
Y0
Z1
[2
\3
]4
^5
_6
`7
a8
b9
c10
d11
e12
f13
g14
h15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_transformer_encoder_25_layer_call_fn_503783
7__inference_transformer_encoder_25_layer_call_fn_503820?
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
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_503947
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_504095?
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
	agamma
bbeta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	cgamma
dbeta"
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
+?&call_and_return_all_conditional_losses

ekernel
fbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

gkernel
hbias"
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
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15"
trackable_list_wrapper
?
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
7__inference_transformer_decoder_25_layer_call_fn_504132
7__inference_transformer_decoder_25_layer_call_fn_504169?
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
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504342
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504536?
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
	qgamma
rbeta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	sgamma
tbeta"
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

ukernel
vbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

wkernel
xbias"
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
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
;__inference_global_average_pooling1d_9_layer_call_fn_504541?
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
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_504547?
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
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_14_layer_call_fn_504556?
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
D__inference_dense_14_layer_call_and_return_conditional_losses_504567?
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
!:  2dense_14/kernel
: 2dense_14/bias
O:M	? 2<token_and_position_embedding_23/token_embedding24/embeddings
Q:O  2?token_and_position_embedding_23/position_embedding24/embeddings
N:L 2<token_and_position_embedding_24/token_embedding25/embeddings
Q:O  2?token_and_position_embedding_24/position_embedding25/embeddings
N:L 28transformer_encoder_25/multi_head_attention/query/kernel
H:F26transformer_encoder_25/multi_head_attention/query/bias
L:J 26transformer_encoder_25/multi_head_attention/key/kernel
F:D24transformer_encoder_25/multi_head_attention/key/bias
N:L 28transformer_encoder_25/multi_head_attention/value/kernel
H:F26transformer_encoder_25/multi_head_attention/value/bias
Y:W 2Ctransformer_encoder_25/multi_head_attention/attention_output/kernel
O:M 2Atransformer_encoder_25/multi_head_attention/attention_output/bias
>:< 20transformer_encoder_25/layer_normalization/gamma
=:; 2/transformer_encoder_25/layer_normalization/beta
@:> 22transformer_encoder_25/layer_normalization_1/gamma
?:= 21transformer_encoder_25/layer_normalization_1/beta
5:3 @2#transformer_encoder_25/dense/kernel
/:-@2!transformer_encoder_25/dense/bias
7:5@ 2%transformer_encoder_25/dense_1/kernel
1:/ 2#transformer_encoder_25/dense_1/bias
N:L 28transformer_decoder_25/multi_head_attention/query/kernel
H:F26transformer_decoder_25/multi_head_attention/query/bias
L:J 26transformer_decoder_25/multi_head_attention/key/kernel
F:D24transformer_decoder_25/multi_head_attention/key/bias
N:L 28transformer_decoder_25/multi_head_attention/value/kernel
H:F26transformer_decoder_25/multi_head_attention/value/bias
Y:W 2Ctransformer_decoder_25/multi_head_attention/attention_output/kernel
O:M 2Atransformer_decoder_25/multi_head_attention/attention_output/bias
>:< 20transformer_decoder_25/layer_normalization/gamma
=:; 2/transformer_decoder_25/layer_normalization/beta
@:> 22transformer_decoder_25/layer_normalization_1/gamma
?:= 21transformer_decoder_25/layer_normalization_1/beta
5:3 @2#transformer_decoder_25/dense/kernel
/:-@2!transformer_decoder_25/dense/bias
7:5@ 2%transformer_decoder_25/dense_1/kernel
1:/ 2#transformer_decoder_25/dense_1/bias
 "
trackable_list_wrapper
f
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
9"
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
(__inference_model_9_layer_call_fn_501282phrase
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
(__inference_model_9_layer_call_fn_502725inputs/0inputs/1"?
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
(__inference_model_9_layer_call_fn_502815inputs/0inputs/1"?
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
(__inference_model_9_layer_call_fn_502217phrase
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
C__inference_model_9_layer_call_and_return_conditional_losses_503217inputs/0inputs/1"?
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
C__inference_model_9_layer_call_and_return_conditional_losses_503661inputs/0inputs/1"?
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
C__inference_model_9_layer_call_and_return_conditional_losses_502353phrase
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
C__inference_model_9_layer_call_and_return_conditional_losses_502489phrase
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
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
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
$__inference_signature_wrapper_502587phrase
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
__inference_adapt_step_502635iterator"?
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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
@__inference_token_and_position_embedding_23_layer_call_fn_503670inputs"?
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
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_503697inputs"?
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
U0"
trackable_list_wrapper
'
U0"
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
V0"
trackable_list_wrapper
'
V0"
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
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
@__inference_token_and_position_embedding_24_layer_call_fn_503706inputs"?
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
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_503734inputs"?
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
W0"
trackable_list_wrapper
'
W0"
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
X0"
trackable_list_wrapper
'
X0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_add_9_layer_call_fn_503740inputs/0inputs/1"?
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
A__inference_add_9_layer_call_and_return_conditional_losses_503746inputs/0inputs/1"?
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
30
41
52
63
74
85
96"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_transformer_encoder_25_layer_call_fn_503783inputs"?
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
7__inference_transformer_encoder_25_layer_call_fn_503820inputs"?
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
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_503947inputs"?
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
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_504095inputs"?
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
Y0
Z1
[2
\3
]4
^5
_6
`7"
trackable_list_wrapper
X
Y0
Z1
[2
\3
]4
^5
_6
`7"
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

Ykernel
Zbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

[kernel
\bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

]kernel
^bias"
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

_kernel
`bias"
_tf_keras_layer
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
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
c0
d1"
trackable_list_wrapper
.
c0
d1"
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
e0
f1"
trackable_list_wrapper
.
e0
f1"
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
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
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
 "
trackable_list_wrapper
Q
@0
A1
B2
C3
D4
E5
F6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
7__inference_transformer_decoder_25_layer_call_fn_504132decoder_sequence"?
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
7__inference_transformer_decoder_25_layer_call_fn_504169decoder_sequence"?
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
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504342decoder_sequence"?
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
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504536decoder_sequence"?
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
X
i0
j1
k2
l3
m4
n5
o6
p7"
trackable_list_wrapper
X
i0
j1
k2
l3
m4
n5
o6
p7"
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

ikernel
jbias"
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

kkernel
lbias"
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

mkernel
nbias"
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

okernel
pbias"
_tf_keras_layer
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
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
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
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
.
u0
v1"
trackable_list_wrapper
.
u0
v1"
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
.
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
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
;__inference_global_average_pooling1d_9_layer_call_fn_504541inputs"?
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
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_504547inputs"?
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
)__inference_dense_14_layer_call_fn_504556inputs"?
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
D__inference_dense_14_layer_call_and_return_conditional_losses_504567inputs"?
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
__inference__creator_504572?
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
__inference__initializer_504580?
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
__inference__destroyer_504585?
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
__inference__creator_504590?
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
__inference__initializer_504595?
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
__inference__destroyer_504600?
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
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
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
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
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
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
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
_0
`1"
trackable_list_wrapper
.
_0
`1"
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
i0
j1"
trackable_list_wrapper
.
i0
j1"
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
k0
l1"
trackable_list_wrapper
.
k0
l1"
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
m0
n1"
trackable_list_wrapper
.
m0
n1"
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
o0
p1"
trackable_list_wrapper
.
o0
p1"
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
__inference__creator_504572"?
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
__inference__initializer_504580"?
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
__inference__destroyer_504585"?
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
__inference__creator_504590"?
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
__inference__initializer_504595"?
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
__inference__destroyer_504600"?
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
&:$  2Adam/dense_14/kernel/m
 : 2Adam/dense_14/bias/m
T:R	? 2CAdam/token_and_position_embedding_23/token_embedding24/embeddings/m
V:T  2FAdam/token_and_position_embedding_23/position_embedding24/embeddings/m
S:Q 2CAdam/token_and_position_embedding_24/token_embedding25/embeddings/m
V:T  2FAdam/token_and_position_embedding_24/position_embedding25/embeddings/m
S:Q 2?Adam/transformer_encoder_25/multi_head_attention/query/kernel/m
M:K2=Adam/transformer_encoder_25/multi_head_attention/query/bias/m
Q:O 2=Adam/transformer_encoder_25/multi_head_attention/key/kernel/m
K:I2;Adam/transformer_encoder_25/multi_head_attention/key/bias/m
S:Q 2?Adam/transformer_encoder_25/multi_head_attention/value/kernel/m
M:K2=Adam/transformer_encoder_25/multi_head_attention/value/bias/m
^:\ 2JAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/m
T:R 2HAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/m
C:A 27Adam/transformer_encoder_25/layer_normalization/gamma/m
B:@ 26Adam/transformer_encoder_25/layer_normalization/beta/m
E:C 29Adam/transformer_encoder_25/layer_normalization_1/gamma/m
D:B 28Adam/transformer_encoder_25/layer_normalization_1/beta/m
::8 @2*Adam/transformer_encoder_25/dense/kernel/m
4:2@2(Adam/transformer_encoder_25/dense/bias/m
<::@ 2,Adam/transformer_encoder_25/dense_1/kernel/m
6:4 2*Adam/transformer_encoder_25/dense_1/bias/m
S:Q 2?Adam/transformer_decoder_25/multi_head_attention/query/kernel/m
M:K2=Adam/transformer_decoder_25/multi_head_attention/query/bias/m
Q:O 2=Adam/transformer_decoder_25/multi_head_attention/key/kernel/m
K:I2;Adam/transformer_decoder_25/multi_head_attention/key/bias/m
S:Q 2?Adam/transformer_decoder_25/multi_head_attention/value/kernel/m
M:K2=Adam/transformer_decoder_25/multi_head_attention/value/bias/m
^:\ 2JAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/m
T:R 2HAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/m
C:A 27Adam/transformer_decoder_25/layer_normalization/gamma/m
B:@ 26Adam/transformer_decoder_25/layer_normalization/beta/m
E:C 29Adam/transformer_decoder_25/layer_normalization_1/gamma/m
D:B 28Adam/transformer_decoder_25/layer_normalization_1/beta/m
::8 @2*Adam/transformer_decoder_25/dense/kernel/m
4:2@2(Adam/transformer_decoder_25/dense/bias/m
<::@ 2,Adam/transformer_decoder_25/dense_1/kernel/m
6:4 2*Adam/transformer_decoder_25/dense_1/bias/m
&:$  2Adam/dense_14/kernel/v
 : 2Adam/dense_14/bias/v
T:R	? 2CAdam/token_and_position_embedding_23/token_embedding24/embeddings/v
V:T  2FAdam/token_and_position_embedding_23/position_embedding24/embeddings/v
S:Q 2CAdam/token_and_position_embedding_24/token_embedding25/embeddings/v
V:T  2FAdam/token_and_position_embedding_24/position_embedding25/embeddings/v
S:Q 2?Adam/transformer_encoder_25/multi_head_attention/query/kernel/v
M:K2=Adam/transformer_encoder_25/multi_head_attention/query/bias/v
Q:O 2=Adam/transformer_encoder_25/multi_head_attention/key/kernel/v
K:I2;Adam/transformer_encoder_25/multi_head_attention/key/bias/v
S:Q 2?Adam/transformer_encoder_25/multi_head_attention/value/kernel/v
M:K2=Adam/transformer_encoder_25/multi_head_attention/value/bias/v
^:\ 2JAdam/transformer_encoder_25/multi_head_attention/attention_output/kernel/v
T:R 2HAdam/transformer_encoder_25/multi_head_attention/attention_output/bias/v
C:A 27Adam/transformer_encoder_25/layer_normalization/gamma/v
B:@ 26Adam/transformer_encoder_25/layer_normalization/beta/v
E:C 29Adam/transformer_encoder_25/layer_normalization_1/gamma/v
D:B 28Adam/transformer_encoder_25/layer_normalization_1/beta/v
::8 @2*Adam/transformer_encoder_25/dense/kernel/v
4:2@2(Adam/transformer_encoder_25/dense/bias/v
<::@ 2,Adam/transformer_encoder_25/dense_1/kernel/v
6:4 2*Adam/transformer_encoder_25/dense_1/bias/v
S:Q 2?Adam/transformer_decoder_25/multi_head_attention/query/kernel/v
M:K2=Adam/transformer_decoder_25/multi_head_attention/query/bias/v
Q:O 2=Adam/transformer_decoder_25/multi_head_attention/key/kernel/v
K:I2;Adam/transformer_decoder_25/multi_head_attention/key/bias/v
S:Q 2?Adam/transformer_decoder_25/multi_head_attention/value/kernel/v
M:K2=Adam/transformer_decoder_25/multi_head_attention/value/bias/v
^:\ 2JAdam/transformer_decoder_25/multi_head_attention/attention_output/kernel/v
T:R 2HAdam/transformer_decoder_25/multi_head_attention/attention_output/bias/v
C:A 27Adam/transformer_decoder_25/layer_normalization/gamma/v
B:@ 26Adam/transformer_decoder_25/layer_normalization/beta/v
E:C 29Adam/transformer_decoder_25/layer_normalization_1/gamma/v
D:B 28Adam/transformer_decoder_25/layer_normalization_1/beta/v
::8 @2*Adam/transformer_decoder_25/dense/kernel/v
4:2@2(Adam/transformer_decoder_25/dense/bias/v
<::@ 2,Adam/transformer_decoder_25/dense_1/kernel/v
6:4 2*Adam/transformer_decoder_25/dense_1/bias/v
?B?
__inference_save_fn_504619checkpoint_key"?
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
__inference_restore_fn_504628restored_tensors_0restored_tensors_1"?
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
__inference__creator_504572?

? 
? "? 7
__inference__creator_504590?

? 
? "? 9
__inference__destroyer_504585?

? 
? "? 9
__inference__destroyer_504600?

? 
? "? C
__inference__initializer_504580 ????

? 
? "? ;
__inference__initializer_504595?

? 
? "? ?
!__inference__wrapped_model_500663?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTZ?W
P?M
K?H
 ?
phrase?????????
$?!

token_role????????? 
? "3?0
.
dense_14"?
dense_14????????? p
__inference_adapt_step_502635O??C?@
9?6
4?1?
??????????IteratorSpec 
? "
 ?
A__inference_add_9_layer_call_and_return_conditional_losses_503746?b?_
X?U
S?P
&?#
inputs/0?????????  
&?#
inputs/1?????????  
? ")?&
?
0?????????  
? ?
&__inference_add_9_layer_call_fn_503740?b?_
X?U
S?P
&?#
inputs/0?????????  
&?#
inputs/1?????????  
? "??????????  ?
D__inference_dense_14_layer_call_and_return_conditional_losses_504567\ST/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? |
)__inference_dense_14_layer_call_fn_504556OST/?,
%?"
 ?
inputs????????? 
? "?????????? ?
V__inference_global_average_pooling1d_9_layer_call_and_return_conditional_losses_504547{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
;__inference_global_average_pooling1d_9_layer_call_fn_504541nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
C__inference_model_9_layer_call_and_return_conditional_losses_502353?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
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
0????????? 
? ?
C__inference_model_9_layer_call_and_return_conditional_losses_502489?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
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
0????????? 
? ?
C__inference_model_9_layer_call_and_return_conditional_losses_503217?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
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
0????????? 
? ?
C__inference_model_9_layer_call_and_return_conditional_losses_503661?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
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
0????????? 
? ?
(__inference_model_9_layer_call_fn_501282?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
X?U
K?H
 ?
phrase?????????
$?!

token_role????????? 
p 

 
? "?????????? ?
(__inference_model_9_layer_call_fn_502217?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
X?U
K?H
 ?
phrase?????????
$?!

token_role????????? 
p

 
? "?????????? ?
(__inference_model_9_layer_call_fn_502725?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
p 

 
? "?????????? ?
(__inference_model_9_layer_call_fn_502815?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTb?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
p

 
? "?????????? {
__inference_restore_fn_504628Z?K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_504619??&?#
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
1/tensor	?
$__inference_signature_wrapper_502587?.????UVWXYZ[\]^_`abefghcdijklmnopqruvwxstSTm?j
? 
c?`
*
phrase ?
phrase?????????
2

token_role$?!

token_role????????? "3?0
.
dense_14"?
dense_14????????? ?
[__inference_token_and_position_embedding_23_layer_call_and_return_conditional_losses_503697`UV/?,
%?"
 ?
inputs????????? 	
? ")?&
?
0?????????  
? ?
@__inference_token_and_position_embedding_23_layer_call_fn_503670SUV/?,
%?"
 ?
inputs????????? 	
? "??????????  ?
[__inference_token_and_position_embedding_24_layer_call_and_return_conditional_losses_503734`WX/?,
%?"
 ?
inputs????????? 
? ")?&
?
0?????????  
? ?
@__inference_token_and_position_embedding_24_layer_call_fn_503706SWX/?,
%?"
 ?
inputs????????? 
? "??????????  ?
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504342?ijklmnopqruvwxsta?^
G?D
.?+
decoder_sequence?????????  
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
0?????????  
? ?
R__inference_transformer_decoder_25_layer_call_and_return_conditional_losses_504536?ijklmnopqruvwxsta?^
G?D
.?+
decoder_sequence?????????  
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
0?????????  
? ?
7__inference_transformer_decoder_25_layer_call_fn_504132?ijklmnopqruvwxsta?^
G?D
.?+
decoder_sequence?????????  
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
trainingp "??????????  ?
7__inference_transformer_decoder_25_layer_call_fn_504169?ijklmnopqruvwxsta?^
G?D
.?+
decoder_sequence?????????  
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
trainingp"??????????  ?
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_503947?YZ[\]^_`abefghcdK?H
1?.
$?!
inputs?????????  

 

 
?

trainingp ")?&
?
0?????????  
? ?
R__inference_transformer_encoder_25_layer_call_and_return_conditional_losses_504095?YZ[\]^_`abefghcdK?H
1?.
$?!
inputs?????????  

 

 
?

trainingp")?&
?
0?????????  
? ?
7__inference_transformer_encoder_25_layer_call_fn_503783}YZ[\]^_`abefghcdK?H
1?.
$?!
inputs?????????  

 

 
?

trainingp "??????????  ?
7__inference_transformer_encoder_25_layer_call_fn_503820}YZ[\]^_`abefghcdK?H
1?.
$?!
inputs?????????  

 

 
?

trainingp"??????????  