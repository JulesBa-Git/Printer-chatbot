??8
?/?.
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??3
?
)Adam/transformer_decoder_3/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/transformer_decoder_3/dense_1/bias/v
?
=Adam/transformer_decoder_3/dense_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/transformer_decoder_3/dense_1/bias/v*
_output_shapes
:*
dtype0
?
+Adam/transformer_decoder_3/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+Adam/transformer_decoder_3/dense_1/kernel/v
?
?Adam/transformer_decoder_3/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/transformer_decoder_3/dense_1/kernel/v*
_output_shapes

:*
dtype0
?
'Adam/transformer_decoder_3/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/transformer_decoder_3/dense/bias/v
?
;Adam/transformer_decoder_3/dense/bias/v/Read/ReadVariableOpReadVariableOp'Adam/transformer_decoder_3/dense/bias/v*
_output_shapes
:*
dtype0
?
)Adam/transformer_decoder_3/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/transformer_decoder_3/dense/kernel/v
?
=Adam/transformer_decoder_3/dense/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/transformer_decoder_3/dense/kernel/v*
_output_shapes

:*
dtype0
?
7Adam/transformer_decoder_3/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_decoder_3/layer_normalization_1/beta/v
?
KAdam/transformer_decoder_3/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_decoder_3/layer_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
8Adam/transformer_decoder_3/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_decoder_3/layer_normalization_1/gamma/v
?
LAdam/transformer_decoder_3/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_3/layer_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
5Adam/transformer_decoder_3/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_decoder_3/layer_normalization/beta/v
?
IAdam/transformer_decoder_3/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_decoder_3/layer_normalization/beta/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_decoder_3/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_decoder_3/layer_normalization/gamma/v
?
JAdam/transformer_decoder_3/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_3/layer_normalization/gamma/v*
_output_shapes
:*
dtype0
?
GAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/v
?
[Adam/transformer_decoder_3/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0
?
IAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/v
?
]Adam/transformer_decoder_3/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
?
<Adam/transformer_decoder_3/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_decoder_3/multi_head_attention/value/bias/v
?
PAdam/transformer_decoder_3/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_decoder_3/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
?
>Adam/transformer_decoder_3/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_decoder_3/multi_head_attention/value/kernel/v
?
RAdam/transformer_decoder_3/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_decoder_3/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
?
:Adam/transformer_decoder_3/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Adam/transformer_decoder_3/multi_head_attention/key/bias/v
?
NAdam/transformer_decoder_3/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_decoder_3/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
?
<Adam/transformer_decoder_3/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_decoder_3/multi_head_attention/key/kernel/v
?
PAdam/transformer_decoder_3/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_decoder_3/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
?
<Adam/transformer_decoder_3/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_decoder_3/multi_head_attention/query/bias/v
?
PAdam/transformer_decoder_3/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_decoder_3/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
?
>Adam/transformer_decoder_3/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_decoder_3/multi_head_attention/query/kernel/v
?
RAdam/transformer_decoder_3/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_decoder_3/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
?
)Adam/transformer_encoder_3/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/transformer_encoder_3/dense_1/bias/v
?
=Adam/transformer_encoder_3/dense_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/transformer_encoder_3/dense_1/bias/v*
_output_shapes
:*
dtype0
?
+Adam/transformer_encoder_3/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+Adam/transformer_encoder_3/dense_1/kernel/v
?
?Adam/transformer_encoder_3/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/transformer_encoder_3/dense_1/kernel/v*
_output_shapes

:*
dtype0
?
'Adam/transformer_encoder_3/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/transformer_encoder_3/dense/bias/v
?
;Adam/transformer_encoder_3/dense/bias/v/Read/ReadVariableOpReadVariableOp'Adam/transformer_encoder_3/dense/bias/v*
_output_shapes
:*
dtype0
?
)Adam/transformer_encoder_3/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/transformer_encoder_3/dense/kernel/v
?
=Adam/transformer_encoder_3/dense/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/transformer_encoder_3/dense/kernel/v*
_output_shapes

:*
dtype0
?
7Adam/transformer_encoder_3/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_encoder_3/layer_normalization_1/beta/v
?
KAdam/transformer_encoder_3/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_encoder_3/layer_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
8Adam/transformer_encoder_3/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_encoder_3/layer_normalization_1/gamma/v
?
LAdam/transformer_encoder_3/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_encoder_3/layer_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
5Adam/transformer_encoder_3/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_encoder_3/layer_normalization/beta/v
?
IAdam/transformer_encoder_3/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_encoder_3/layer_normalization/beta/v*
_output_shapes
:*
dtype0
?
6Adam/transformer_encoder_3/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_encoder_3/layer_normalization/gamma/v
?
JAdam/transformer_encoder_3/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_encoder_3/layer_normalization/gamma/v*
_output_shapes
:*
dtype0
?
GAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/v
?
[Adam/transformer_encoder_3/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/v*
_output_shapes
:*
dtype0
?
IAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/v
?
]Adam/transformer_encoder_3/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:*
dtype0
?
<Adam/transformer_encoder_3/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_encoder_3/multi_head_attention/value/bias/v
?
PAdam/transformer_encoder_3/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_encoder_3/multi_head_attention/value/bias/v*
_output_shapes

:*
dtype0
?
>Adam/transformer_encoder_3/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_encoder_3/multi_head_attention/value/kernel/v
?
RAdam/transformer_encoder_3/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_encoder_3/multi_head_attention/value/kernel/v*"
_output_shapes
:*
dtype0
?
:Adam/transformer_encoder_3/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Adam/transformer_encoder_3/multi_head_attention/key/bias/v
?
NAdam/transformer_encoder_3/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp:Adam/transformer_encoder_3/multi_head_attention/key/bias/v*
_output_shapes

:*
dtype0
?
<Adam/transformer_encoder_3/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_encoder_3/multi_head_attention/key/kernel/v
?
PAdam/transformer_encoder_3/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_encoder_3/multi_head_attention/key/kernel/v*"
_output_shapes
:*
dtype0
?
<Adam/transformer_encoder_3/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_encoder_3/multi_head_attention/query/bias/v
?
PAdam/transformer_encoder_3/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp<Adam/transformer_encoder_3/multi_head_attention/query/bias/v*
_output_shapes

:*
dtype0
?
>Adam/transformer_encoder_3/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_encoder_3/multi_head_attention/query/kernel/v
?
RAdam/transformer_encoder_3/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp>Adam/transformer_encoder_3/multi_head_attention/query/kernel/v*"
_output_shapes
:*
dtype0
?
DAdam/token_and_position_embedding_2/position_embedding3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDAdam/token_and_position_embedding_2/position_embedding3/embeddings/v
?
XAdam/token_and_position_embedding_2/position_embedding3/embeddings/v/Read/ReadVariableOpReadVariableOpDAdam/token_and_position_embedding_2/position_embedding3/embeddings/v*
_output_shapes

: *
dtype0
?
AAdam/token_and_position_embedding_2/token_embedding3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*R
shared_nameCAAdam/token_and_position_embedding_2/token_embedding3/embeddings/v
?
UAdam/token_and_position_embedding_2/token_embedding3/embeddings/v/Read/ReadVariableOpReadVariableOpAAdam/token_and_position_embedding_2/token_embedding3/embeddings/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:@ *
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_3/embeddings/v
?
1Adam/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_3/embeddings/v*
_output_shapes

:*
dtype0
?
)Adam/transformer_decoder_3/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/transformer_decoder_3/dense_1/bias/m
?
=Adam/transformer_decoder_3/dense_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/transformer_decoder_3/dense_1/bias/m*
_output_shapes
:*
dtype0
?
+Adam/transformer_decoder_3/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+Adam/transformer_decoder_3/dense_1/kernel/m
?
?Adam/transformer_decoder_3/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/transformer_decoder_3/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
'Adam/transformer_decoder_3/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/transformer_decoder_3/dense/bias/m
?
;Adam/transformer_decoder_3/dense/bias/m/Read/ReadVariableOpReadVariableOp'Adam/transformer_decoder_3/dense/bias/m*
_output_shapes
:*
dtype0
?
)Adam/transformer_decoder_3/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/transformer_decoder_3/dense/kernel/m
?
=Adam/transformer_decoder_3/dense/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/transformer_decoder_3/dense/kernel/m*
_output_shapes

:*
dtype0
?
7Adam/transformer_decoder_3/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_decoder_3/layer_normalization_1/beta/m
?
KAdam/transformer_decoder_3/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_decoder_3/layer_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
8Adam/transformer_decoder_3/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_decoder_3/layer_normalization_1/gamma/m
?
LAdam/transformer_decoder_3/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_decoder_3/layer_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
5Adam/transformer_decoder_3/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_decoder_3/layer_normalization/beta/m
?
IAdam/transformer_decoder_3/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_decoder_3/layer_normalization/beta/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_decoder_3/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_decoder_3/layer_normalization/gamma/m
?
JAdam/transformer_decoder_3/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_decoder_3/layer_normalization/gamma/m*
_output_shapes
:*
dtype0
?
GAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/m
?
[Adam/transformer_decoder_3/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0
?
IAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/m
?
]Adam/transformer_decoder_3/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
?
<Adam/transformer_decoder_3/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_decoder_3/multi_head_attention/value/bias/m
?
PAdam/transformer_decoder_3/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_decoder_3/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
?
>Adam/transformer_decoder_3/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_decoder_3/multi_head_attention/value/kernel/m
?
RAdam/transformer_decoder_3/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_decoder_3/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
?
:Adam/transformer_decoder_3/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Adam/transformer_decoder_3/multi_head_attention/key/bias/m
?
NAdam/transformer_decoder_3/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_decoder_3/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
?
<Adam/transformer_decoder_3/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_decoder_3/multi_head_attention/key/kernel/m
?
PAdam/transformer_decoder_3/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_decoder_3/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
?
<Adam/transformer_decoder_3/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_decoder_3/multi_head_attention/query/bias/m
?
PAdam/transformer_decoder_3/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_decoder_3/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
?
>Adam/transformer_decoder_3/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_decoder_3/multi_head_attention/query/kernel/m
?
RAdam/transformer_decoder_3/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_decoder_3/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
?
)Adam/transformer_encoder_3/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/transformer_encoder_3/dense_1/bias/m
?
=Adam/transformer_encoder_3/dense_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/transformer_encoder_3/dense_1/bias/m*
_output_shapes
:*
dtype0
?
+Adam/transformer_encoder_3/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+Adam/transformer_encoder_3/dense_1/kernel/m
?
?Adam/transformer_encoder_3/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/transformer_encoder_3/dense_1/kernel/m*
_output_shapes

:*
dtype0
?
'Adam/transformer_encoder_3/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/transformer_encoder_3/dense/bias/m
?
;Adam/transformer_encoder_3/dense/bias/m/Read/ReadVariableOpReadVariableOp'Adam/transformer_encoder_3/dense/bias/m*
_output_shapes
:*
dtype0
?
)Adam/transformer_encoder_3/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*:
shared_name+)Adam/transformer_encoder_3/dense/kernel/m
?
=Adam/transformer_encoder_3/dense/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/transformer_encoder_3/dense/kernel/m*
_output_shapes

:*
dtype0
?
7Adam/transformer_encoder_3/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/transformer_encoder_3/layer_normalization_1/beta/m
?
KAdam/transformer_encoder_3/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_encoder_3/layer_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
8Adam/transformer_encoder_3/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/transformer_encoder_3/layer_normalization_1/gamma/m
?
LAdam/transformer_encoder_3/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_encoder_3/layer_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
5Adam/transformer_encoder_3/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/transformer_encoder_3/layer_normalization/beta/m
?
IAdam/transformer_encoder_3/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_encoder_3/layer_normalization/beta/m*
_output_shapes
:*
dtype0
?
6Adam/transformer_encoder_3/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*G
shared_name86Adam/transformer_encoder_3/layer_normalization/gamma/m
?
JAdam/transformer_encoder_3/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_encoder_3/layer_normalization/gamma/m*
_output_shapes
:*
dtype0
?
GAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*X
shared_nameIGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/m
?
[Adam/transformer_encoder_3/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/m*
_output_shapes
:*
dtype0
?
IAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Z
shared_nameKIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/m
?
]Adam/transformer_encoder_3/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:*
dtype0
?
<Adam/transformer_encoder_3/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_encoder_3/multi_head_attention/value/bias/m
?
PAdam/transformer_encoder_3/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_encoder_3/multi_head_attention/value/bias/m*
_output_shapes

:*
dtype0
?
>Adam/transformer_encoder_3/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_encoder_3/multi_head_attention/value/kernel/m
?
RAdam/transformer_encoder_3/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_encoder_3/multi_head_attention/value/kernel/m*"
_output_shapes
:*
dtype0
?
:Adam/transformer_encoder_3/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*K
shared_name<:Adam/transformer_encoder_3/multi_head_attention/key/bias/m
?
NAdam/transformer_encoder_3/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp:Adam/transformer_encoder_3/multi_head_attention/key/bias/m*
_output_shapes

:*
dtype0
?
<Adam/transformer_encoder_3/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*M
shared_name><Adam/transformer_encoder_3/multi_head_attention/key/kernel/m
?
PAdam/transformer_encoder_3/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_encoder_3/multi_head_attention/key/kernel/m*"
_output_shapes
:*
dtype0
?
<Adam/transformer_encoder_3/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*M
shared_name><Adam/transformer_encoder_3/multi_head_attention/query/bias/m
?
PAdam/transformer_encoder_3/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp<Adam/transformer_encoder_3/multi_head_attention/query/bias/m*
_output_shapes

:*
dtype0
?
>Adam/transformer_encoder_3/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*O
shared_name@>Adam/transformer_encoder_3/multi_head_attention/query/kernel/m
?
RAdam/transformer_encoder_3/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp>Adam/transformer_encoder_3/multi_head_attention/query/kernel/m*"
_output_shapes
:*
dtype0
?
DAdam/token_and_position_embedding_2/position_embedding3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *U
shared_nameFDAdam/token_and_position_embedding_2/position_embedding3/embeddings/m
?
XAdam/token_and_position_embedding_2/position_embedding3/embeddings/m/Read/ReadVariableOpReadVariableOpDAdam/token_and_position_embedding_2/position_embedding3/embeddings/m*
_output_shapes

: *
dtype0
?
AAdam/token_and_position_embedding_2/token_embedding3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*R
shared_nameCAAdam/token_and_position_embedding_2/token_embedding3/embeddings/m
?
UAdam/token_and_position_embedding_2/token_embedding3/embeddings/m/Read/ReadVariableOpReadVariableOpAAdam/token_and_position_embedding_2/token_embedding3/embeddings/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:@ *
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_3/embeddings/m
?
1Adam/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_3/embeddings/m*
_output_shapes

:*
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
|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
value_dtype0	
n

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name318609*
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
"transformer_decoder_3/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"transformer_decoder_3/dense_1/bias
?
6transformer_decoder_3/dense_1/bias/Read/ReadVariableOpReadVariableOp"transformer_decoder_3/dense_1/bias*
_output_shapes
:*
dtype0
?
$transformer_decoder_3/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$transformer_decoder_3/dense_1/kernel
?
8transformer_decoder_3/dense_1/kernel/Read/ReadVariableOpReadVariableOp$transformer_decoder_3/dense_1/kernel*
_output_shapes

:*
dtype0
?
 transformer_decoder_3/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" transformer_decoder_3/dense/bias
?
4transformer_decoder_3/dense/bias/Read/ReadVariableOpReadVariableOp transformer_decoder_3/dense/bias*
_output_shapes
:*
dtype0
?
"transformer_decoder_3/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"transformer_decoder_3/dense/kernel
?
6transformer_decoder_3/dense/kernel/Read/ReadVariableOpReadVariableOp"transformer_decoder_3/dense/kernel*
_output_shapes

:*
dtype0
?
0transformer_decoder_3/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_decoder_3/layer_normalization_1/beta
?
Dtransformer_decoder_3/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp0transformer_decoder_3/layer_normalization_1/beta*
_output_shapes
:*
dtype0
?
1transformer_decoder_3/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31transformer_decoder_3/layer_normalization_1/gamma
?
Etransformer_decoder_3/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp1transformer_decoder_3/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
?
.transformer_decoder_3/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.transformer_decoder_3/layer_normalization/beta
?
Btransformer_decoder_3/layer_normalization/beta/Read/ReadVariableOpReadVariableOp.transformer_decoder_3/layer_normalization/beta*
_output_shapes
:*
dtype0
?
/transformer_decoder_3/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_decoder_3/layer_normalization/gamma
?
Ctransformer_decoder_3/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp/transformer_decoder_3/layer_normalization/gamma*
_output_shapes
:*
dtype0
?
@transformer_decoder_3/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@transformer_decoder_3/multi_head_attention/attention_output/bias
?
Ttransformer_decoder_3/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_decoder_3/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
?
Btransformer_decoder_3/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBtransformer_decoder_3/multi_head_attention/attention_output/kernel
?
Vtransformer_decoder_3/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_decoder_3/multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
?
5transformer_decoder_3/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75transformer_decoder_3/multi_head_attention/value/bias
?
Itransformer_decoder_3/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp5transformer_decoder_3/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
?
7transformer_decoder_3/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97transformer_decoder_3/multi_head_attention/value/kernel
?
Ktransformer_decoder_3/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_decoder_3/multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0
?
3transformer_decoder_3/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53transformer_decoder_3/multi_head_attention/key/bias
?
Gtransformer_decoder_3/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp3transformer_decoder_3/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
?
5transformer_decoder_3/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_decoder_3/multi_head_attention/key/kernel
?
Itransformer_decoder_3/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_decoder_3/multi_head_attention/key/kernel*"
_output_shapes
:*
dtype0
?
5transformer_decoder_3/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75transformer_decoder_3/multi_head_attention/query/bias
?
Itransformer_decoder_3/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp5transformer_decoder_3/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
?
7transformer_decoder_3/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97transformer_decoder_3/multi_head_attention/query/kernel
?
Ktransformer_decoder_3/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_decoder_3/multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0
?
"transformer_encoder_3/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"transformer_encoder_3/dense_1/bias
?
6transformer_encoder_3/dense_1/bias/Read/ReadVariableOpReadVariableOp"transformer_encoder_3/dense_1/bias*
_output_shapes
:*
dtype0
?
$transformer_encoder_3/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$transformer_encoder_3/dense_1/kernel
?
8transformer_encoder_3/dense_1/kernel/Read/ReadVariableOpReadVariableOp$transformer_encoder_3/dense_1/kernel*
_output_shapes

:*
dtype0
?
 transformer_encoder_3/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" transformer_encoder_3/dense/bias
?
4transformer_encoder_3/dense/bias/Read/ReadVariableOpReadVariableOp transformer_encoder_3/dense/bias*
_output_shapes
:*
dtype0
?
"transformer_encoder_3/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"transformer_encoder_3/dense/kernel
?
6transformer_encoder_3/dense/kernel/Read/ReadVariableOpReadVariableOp"transformer_encoder_3/dense/kernel*
_output_shapes

:*
dtype0
?
0transformer_encoder_3/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20transformer_encoder_3/layer_normalization_1/beta
?
Dtransformer_encoder_3/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp0transformer_encoder_3/layer_normalization_1/beta*
_output_shapes
:*
dtype0
?
1transformer_encoder_3/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31transformer_encoder_3/layer_normalization_1/gamma
?
Etransformer_encoder_3/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp1transformer_encoder_3/layer_normalization_1/gamma*
_output_shapes
:*
dtype0
?
.transformer_encoder_3/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.transformer_encoder_3/layer_normalization/beta
?
Btransformer_encoder_3/layer_normalization/beta/Read/ReadVariableOpReadVariableOp.transformer_encoder_3/layer_normalization/beta*
_output_shapes
:*
dtype0
?
/transformer_encoder_3/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/transformer_encoder_3/layer_normalization/gamma
?
Ctransformer_encoder_3/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp/transformer_encoder_3/layer_normalization/gamma*
_output_shapes
:*
dtype0
?
@transformer_encoder_3/multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@transformer_encoder_3/multi_head_attention/attention_output/bias
?
Ttransformer_encoder_3/multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_encoder_3/multi_head_attention/attention_output/bias*
_output_shapes
:*
dtype0
?
Btransformer_encoder_3/multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBtransformer_encoder_3/multi_head_attention/attention_output/kernel
?
Vtransformer_encoder_3/multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_encoder_3/multi_head_attention/attention_output/kernel*"
_output_shapes
:*
dtype0
?
5transformer_encoder_3/multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75transformer_encoder_3/multi_head_attention/value/bias
?
Itransformer_encoder_3/multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOp5transformer_encoder_3/multi_head_attention/value/bias*
_output_shapes

:*
dtype0
?
7transformer_encoder_3/multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97transformer_encoder_3/multi_head_attention/value/kernel
?
Ktransformer_encoder_3/multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_encoder_3/multi_head_attention/value/kernel*"
_output_shapes
:*
dtype0
?
3transformer_encoder_3/multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53transformer_encoder_3/multi_head_attention/key/bias
?
Gtransformer_encoder_3/multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOp3transformer_encoder_3/multi_head_attention/key/bias*
_output_shapes

:*
dtype0
?
5transformer_encoder_3/multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75transformer_encoder_3/multi_head_attention/key/kernel
?
Itransformer_encoder_3/multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_encoder_3/multi_head_attention/key/kernel*"
_output_shapes
:*
dtype0
?
5transformer_encoder_3/multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*F
shared_name75transformer_encoder_3/multi_head_attention/query/bias
?
Itransformer_encoder_3/multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOp5transformer_encoder_3/multi_head_attention/query/bias*
_output_shapes

:*
dtype0
?
7transformer_encoder_3/multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97transformer_encoder_3/multi_head_attention/query/kernel
?
Ktransformer_encoder_3/multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_encoder_3/multi_head_attention/query/kernel*"
_output_shapes
:*
dtype0
?
=token_and_position_embedding_2/position_embedding3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=token_and_position_embedding_2/position_embedding3/embeddings
?
Qtoken_and_position_embedding_2/position_embedding3/embeddings/Read/ReadVariableOpReadVariableOp=token_and_position_embedding_2/position_embedding3/embeddings*
_output_shapes

: *
dtype0
?
:token_and_position_embedding_2/token_embedding3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*K
shared_name<:token_and_position_embedding_2/token_embedding3/embeddings
?
Ntoken_and_position_embedding_2/token_embedding3/embeddings/Read/ReadVariableOpReadVariableOp:token_and_position_embedding_2/token_embedding3/embeddings*
_output_shapes
:	?*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@ *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
?
embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_3/embeddings
?
*embedding_3/embeddings/Read/ReadVariableOpReadVariableOpembedding_3/embeddings*
_output_shapes

:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
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
value	B	 R 
?&
Const_4Const*
_output_shapes	
:?*
dtype0*?&
value?&B?&?BpageB
exemplaireBfoisBtirageB
impressionB	souhaiterBéditionBjusqu’BmerciBstpBsiBpossibleBaimerBvouloirBsalutBbonjourBs’BplerBprésentationBdocumentBd’Bj’BéditerBtirerBimprimerBfaireBavanceBfichierBaimeraiBphotoBl’B	commencerBplopBheyBvidéoBdessinBstarterBsauraiBsaiBpouvoirBpeuxBpeuBlancerB	démarrerBcodeBbalancerBarticleBprojetBéditeBimprimeBcopieBplzBhelloBcoucouBcoolBcelaBrapportBpdfBlanceBimageBtireBtirBeffectueBpleaseBfileBdémarreBdémarrBattenteBimprimBt’BthanksB	remercierBmettreBattendreBjeBversionBpapierBzjqjmBzdzqbByrryosbByrbcdtBypwbByivqBxuiwBxckrmtBwsmxrBwfqwBvyghBvqvBvpcmgwBupifBuknhBugzeBtpxcvbaBtovhjBtllwldrgBtljlcBsrjlwBsrcaBshiieBrwqerBrlnqBqlgzBpzuqydzBouwdhkBosmeBorpadBoblwuBnyaoBnryxajBnfzajelBmzfnopBmyfxdjxoBmohyujwBmmkpeBmfucjBlsqvBklbebBkicmrjwBkhnyaxjBjvjldBjoslBjcfvBimfkbouyBijhqkBhzcqbBhyffBhuuiBhnqlwBhbyvisqBgxuoggBgvdmBfwdpzBfprxBfdodhBejowzdBebencqBdvjhrBduitBcvjrBcsfodogBcpfwjwBbuqdlBbtpntBbqplBbhuaBamxeatBakrpBahelbdBzuhuBzriyqBzcrzjlhByucppiBxjlayazBwvruhkmBwghvojlBwaubwdBvuahrwvjBvnrsBvcaiBvbsuBusqbBuchnBuaubfeBtvpBtshmbstBtdqiBswwaBrzmklfiBrynhBroptdfoBrbnfvBqzzvBqzlrBqlesBpgmyBpfymceBnhwqzzobBmkrrsBmkpbhyBmhfjkBlnbwBljjotkvBleviBlerofBkyeafhyBkwpjBkvvxrBkmihdBjujcBjprishBjinqjvBjelqheBhvpgBhllbBhgtlpcBgwemBgtuBgrhaBgqofoerBgnuyBgmkrBgedtpBfvkaBfsuqBevwrBevmwBeusaktBestabfkaBenaeB	domxsqlvlBdjeBdgmgBdeaobsvBczvcoxmBcpnrtyBckuwihzBckhgBckcheBbpohBbpkvBbpbqipBbohhdqBbmhlBbimrbbbBaqahwBapuvyvBfoiBxbeoBtwgtBsqvvBrqzdBrftcBoeagBnqntBmpnkBlshgBhgdmBfzqlBfkdvBzzzy.fgtB	zzzvd.gllBzzxdrB	zzwnl.wlnBzzwbeB	zzswhnlfuBzzruB
zzrct.eccnBzzqlBzzpljpBzzmmqBzzmh.lhnBzzmaBzzks.herBzzkgo.chBzzif.xdgB	zzhco.kffBzzgg.rhB	zzfy.voppB	zzfgpqvscBzzcgdBzzayuyl.lpqBzzapBzzafzgo.nampBzyxfBzyvyBzyvglBzytmvjB
zysxl.scfyBzyqfBzypvcqBzypc.ihBzynkfrBzyndBzylvBzylteBzyjuevd.jynBzyih.ztvB	zyieyx.gdBzyhgbdBzygguBzyfujqtn.ugdoBzyfgBzyekBzyblzaBzybc.nlsB
zxytvh.vguBzxyqBzxyoBzxxzBzxwxucnwB	zxtfv.lsoBzxtc.lrpBzxocwBzxmxeBzxjyzzBzxjerB	zxihm.wdiBzxgq.shBzxflB	zxdwdl.ziBzxccbBzxbhlBzxagtfB
zwzqf.anttBzwyca.efBzwxuBzwxhoBzwurBzwtmghBzwso.znBzwsdBzwpofB
zwpfcx.echBzwmvkB
zwlgeg.uuyBzwgvgBzwekBzwedBzwdfBzwbdBzwbclBzwaoBzwaekowx.nbB	zvxwe.aypBzvvlaBzvvcBzvuyxfBzvuqBzvsyiB	zvrzy.fueBzvrztBzvqsozBzvpx.jhfBzvmjipBzvlzzcBzvlwB	zvkpw.pctBzvkhBzvkdBzvkbox.zaqerBzvjuBzvhcBzvhBzvgldkBzvgke.fcBzvftm.ekBzvfhdBzvelmka.axgBzvekBzvdeBzvcizzB	zvbu.owxdBzvbkhfB	zuzyz.nzxBzuzuljB	zuxum.qjqBzuxhwdn.huiBzuoBzummuog.zxqwBzulr.hrvBzujhkbBzuiyclyBzuiqtBzuhyBzugaBzudtlB	zucu.frprBzubcBzubaxaB	zuavr.pkvBztznetBztxqdBztwmBztuujr.wvnjBztuln.dkszbjBzttzhdenBztqvB
ztqblsr.onBztpl.foBztpb.zdvBztno.wknB
ztnfxej.bxBztmrkBztmkwiBztjamqBztgfBztfeBztemBztcn.fasB
ztaqbn.kzfBztaisgBzszvBzszkBzsziwkgBzsykdjB	zsxmimoxbB
zswfxl.imvBzsrlBzsnl.jB
zsmmx.ifxnBzslobdBzslcbeBzsla.hsuBzskhBzshaBzsggdBzsgflwBzscgqBzsbhBzsawm.nnBzsaeman.bhbBzsabBzrzdjBzrym.ozpBzrxzjv.fvhgBzrwrpmyBzrwbB
zrvulo.nzbB
zrvgy.zxqpBzrvf.vujBzrvb.ucBzrrx.lshBzrqBzrpwBzrpc.mwdBzrnzzB	zrmew.hrgBzrlf.atBzrkkwwmB
zrhyp.xuqlBzrgotfBzrgdsmmkBzrfpB	zrfe.ootcBzrejwBzrdfvrBzrdbBzrcputBzrcp.cgfBzrckaBzrcBzrbbci.mxvzBzqzkwlja.dfhBzqzjgbcBzqybp.coBzqxrerBzqxg.fjpBzqwedbm.qmdvBzqvw.zoBzqufiBzqudbhff.buB
zqsewp.dduBzqre.sknBzqqcnhkk.rpBzqpxfvBzqmiro.haibB	zqkuj.usfBzqkjfevB
zqjzz.zphxB	zqjxfv.yzB
zqgxd.jsfxBzqgihBzqgdkBzqfqte.hnwdBzqeznxB
zqcwvl.edgB	zqcpu.tmqB
zqcpspqqmxBzqbxBzqaaewBzqaBzpzpepB
zpyrhm.lalBzpypaxyBzpyckBzpwouraBzpwgptBzpvnfllBzpvdavcBzpsl.jsaBzppyBzpptfhBzpprrvBzppfoBzppc.ostBzpojBzpoaxBzplv.bqxBzplrBzpjyB	zpjtqe.tbB
zpjmw.pojnBzpjBzphiB	zpfsdf.wbB
zpfqn.zmxtBzpdfyBzpbkjrng.tyaB	zpao.tpqsB	zozii.nzpBzoyqBzoxhwBzovqB
zouryy.aiqBzoszbBzosp.zpzBzoogbmBzonooBzomwoxBzokpBzojmlqB
zohwoja.luB	zohti.mcwBzoekBzoei.vbjBzocumaoBzobjBznyj.loBznxkB
zntlwc.upwB	znqwe.ijlB	znoy.qevtBznobbB	znmhp.iigBznlpizBznkoB	znkhbx.lbBznimBznfwctewBznfoB
znctlz.nhoBznbxxqBznam.wmB	zmyztd.qrB
zmybdz.eajBzmxsams.nriBzmwrpoB	zmvx.tqerBzmvtBzmvndbBzmvhB	zmuec.xtpBzmuawuvBzmtdfnB
zmmfs.hdbuBzmipss.arupBzmiey.tkBzmhyBzmhcfB	zmdjfd.esB	zmcxw.cidBzlzstBzlxeB
zlvpki.frlBzlvc.evjBzlvbfBzltlkB
zltfhnm.juBzltbcpBzlswvfi.lhuBzlshjzB
zlrvds.qsaBzlqgBzlohoB	zlof.slzcBzlnwBzlmmB
zlidm.waeaB
zlgmrs.qeuBzlcf.frnBzlbz.mqjB
zkzll.bcseBzkxp.zumB
zkwpi.vbpzBzkurBzkuemBzktzBzksmBzksjbBzkqmubvBzkpx.xoBzkoa.kwkBzkjqbxvf.vtlBzkjhBzkhbB
zkfwaa.wbaBzkfd.ioBzkdbzpjBzkcjB
zkbood.qloBzkbfBzkbaBzkabBzjyoB	zjygw.xpzBzjwe.bbjBzjwbe
?&
Const_5Const*
_output_shapes	
:?*
dtype0	*?%
value?%B?%	?"?%                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_336706
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *$
fR
__inference_<lambda>_336711
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
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
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
;
	keras_api
_lookup_layer
_adapt_function*
* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding*
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&
embeddings*
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
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias*
?
]1
^2
&3
_4
`5
a6
b7
c8
d9
e10
f11
g12
h13
i14
j15
k16
l17
m18
n19
o20
p21
q22
r23
s24
t25
u26
v27
w28
x29
y30
z31
{32
|33
}34
~35
S36
T37
[38
\39*
?
]0
^1
&2
_3
`4
a5
b6
c7
d8
e9
f10
g11
h12
i13
j14
k15
l16
m17
n18
o19
p20
q21
r22
s23
t24
u25
v26
w27
x28
y29
z30
{31
|32
}33
~34
S35
T36
[37
\38*
* 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate&m?Sm?Tm?[m?\m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?&v?Sv?Tv?[v?\v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?*
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
]0
^1*

]0
^1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
]
embeddings*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
^
embeddings
^position_embeddings*

&0*

&0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEembedding_3/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
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
_0
`1
a2
b3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15*
z
_0
`1
a2
b3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15*
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
	ggamma
hbeta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	igamma
jbeta*
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

kkernel
lbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

mkernel
nbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator* 
z
o0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15*
z
o0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?	variables
?trainable_variables
?regularization_losses
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
	wgamma
xbeta*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	ygamma
zbeta*
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

{kernel
|bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

}kernel
~bias*
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
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

[0
\1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:token_and_position_embedding_2/token_embedding3/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE=token_and_position_embedding_2/position_embedding3/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE7transformer_encoder_3/multi_head_attention/query/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5transformer_encoder_3/multi_head_attention/query/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5transformer_encoder_3/multi_head_attention/key/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3transformer_encoder_3/multi_head_attention/key/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE7transformer_encoder_3/multi_head_attention/value/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5transformer_encoder_3/multi_head_attention/value/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEBtransformer_encoder_3/multi_head_attention/attention_output/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUE@transformer_encoder_3/multi_head_attention/attention_output/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_encoder_3/layer_normalization/gamma'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.transformer_encoder_3/layer_normalization/beta'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_encoder_3/layer_normalization_1/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_encoder_3/layer_normalization_1/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"transformer_encoder_3/dense/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE transformer_encoder_3/dense/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$transformer_encoder_3/dense_1/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"transformer_encoder_3/dense_1/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_decoder_3/multi_head_attention/query/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_decoder_3/multi_head_attention/query/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_decoder_3/multi_head_attention/key/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3transformer_decoder_3/multi_head_attention/key/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7transformer_decoder_3/multi_head_attention/value/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5transformer_decoder_3/multi_head_attention/value/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEBtransformer_decoder_3/multi_head_attention/attention_output/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUE@transformer_decoder_3/multi_head_attention/attention_output/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/transformer_decoder_3/layer_normalization/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.transformer_decoder_3/layer_normalization/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1transformer_decoder_3/layer_normalization_1/gamma'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0transformer_decoder_3/layer_normalization_1/beta'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"transformer_decoder_3/dense/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE transformer_decoder_3/dense/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$transformer_decoder_3/dense_1/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"transformer_decoder_3/dense_1/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
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
10*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
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
* 
* 
V
?_initializer
?_create_resource
?_initialize
?_destroy_resource* 
?
?_create_resource
?_initialize
?_destroy_resource><layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/*
* 
* 

0
1*
* 
* 
* 
* 
* 

]0*

]0*
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

^0*

^0*
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
_0
`1
a2
b3
c4
d5
e6
f7*
<
_0
`1
a2
b3
c4
d5
e6
f7*
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

_kernel
`bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

akernel
bbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

ckernel
dbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
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

ekernel
fbias*
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

i0
j1*

i0
j1*
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
k0
l1*

k0
l1*
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
m0
n1*

m0
n1*
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
o0
p1
q2
r3
s4
t5
u6
v7*
<
o0
p1
q2
r3
s4
t5
u6
v7*
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

okernel
pbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

qkernel
rbias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?partial_output_shape
?full_output_shape

skernel
tbias*
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

ukernel
vbias*
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
y0
z1*

y0
z1*
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
{0
|1*

{0
|1*
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
}0
~1*

}0
~1*
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
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

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
a0
b1*

a0
b1*
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
c0
d1*

c0
d1*
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
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

q0
r1*

q0
r1*
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
s0
t1*

s0
t1*
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
u0
v1*

u0
v1*
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
?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUEAdam/embedding_3/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/token_and_position_embedding_2/token_embedding3/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDAdam/token_and_position_embedding_2/position_embedding3/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_encoder_3/multi_head_attention/query/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_encoder_3/multi_head_attention/query/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_encoder_3/multi_head_attention/key/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE:Adam/transformer_encoder_3/multi_head_attention/key/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_encoder_3/multi_head_attention/value/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_encoder_3/multi_head_attention/value/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_encoder_3/layer_normalization/gamma/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5Adam/transformer_encoder_3/layer_normalization/beta/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_encoder_3/layer_normalization_1/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_encoder_3/layer_normalization_1/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_encoder_3/dense/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUE'Adam/transformer_encoder_3/dense/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/transformer_encoder_3/dense_1/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_encoder_3/dense_1/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_decoder_3/multi_head_attention/query/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_decoder_3/multi_head_attention/query/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_decoder_3/multi_head_attention/key/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE:Adam/transformer_decoder_3/multi_head_attention/key/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_decoder_3/multi_head_attention/value/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_decoder_3/multi_head_attention/value/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_decoder_3/layer_normalization/gamma/mCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5Adam/transformer_decoder_3/layer_normalization/beta/mCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_decoder_3/layer_normalization_1/gamma/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_decoder_3/layer_normalization_1/beta/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_decoder_3/dense/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUE'Adam/transformer_decoder_3/dense/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/transformer_decoder_3/dense_1/kernel/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_decoder_3/dense_1/bias/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding_3/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAAdam/token_and_position_embedding_2/token_embedding3/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDAdam/token_and_position_embedding_2/position_embedding3/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_encoder_3/multi_head_attention/query/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_encoder_3/multi_head_attention/query/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_encoder_3/multi_head_attention/key/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE:Adam/transformer_encoder_3/multi_head_attention/key/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_encoder_3/multi_head_attention/value/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_encoder_3/multi_head_attention/value/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_encoder_3/layer_normalization/gamma/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5Adam/transformer_encoder_3/layer_normalization/beta/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_encoder_3/layer_normalization_1/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_encoder_3/layer_normalization_1/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_encoder_3/dense/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUE'Adam/transformer_encoder_3/dense/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/transformer_encoder_3/dense_1/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_encoder_3/dense_1/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_decoder_3/multi_head_attention/query/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_decoder_3/multi_head_attention/query/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_decoder_3/multi_head_attention/key/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE:Adam/transformer_decoder_3/multi_head_attention/key/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/transformer_decoder_3/multi_head_attention/value/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE<Adam/transformer_decoder_3/multi_head_attention/value/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/transformer_decoder_3/layer_normalization/gamma/vCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5Adam/transformer_decoder_3/layer_normalization/beta/vCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/transformer_decoder_3/layer_normalization_1/gamma/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7Adam/transformer_decoder_3/layer_normalization_1/beta/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_decoder_3/dense/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUE'Adam/transformer_decoder_3/dense/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE+Adam/transformer_decoder_3/dense_1/kernel/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE)Adam/transformer_decoder_3/dense_1/bias/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
serving_default_PhrasePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_Token_rolePlaceholder*'
_output_shapes
:????????? *
dtype0*
shape:????????? 
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_Phraseserving_default_Token_role
hash_tableConstConst_1Const_2:token_and_position_embedding_2/token_embedding3/embeddings=token_and_position_embedding_2/position_embedding3/embeddingsembedding_3/embeddings7transformer_encoder_3/multi_head_attention/query/kernel5transformer_encoder_3/multi_head_attention/query/bias5transformer_encoder_3/multi_head_attention/key/kernel3transformer_encoder_3/multi_head_attention/key/bias7transformer_encoder_3/multi_head_attention/value/kernel5transformer_encoder_3/multi_head_attention/value/biasBtransformer_encoder_3/multi_head_attention/attention_output/kernel@transformer_encoder_3/multi_head_attention/attention_output/bias/transformer_encoder_3/layer_normalization/gamma.transformer_encoder_3/layer_normalization/beta"transformer_encoder_3/dense/kernel transformer_encoder_3/dense/bias$transformer_encoder_3/dense_1/kernel"transformer_encoder_3/dense_1/bias1transformer_encoder_3/layer_normalization_1/gamma0transformer_encoder_3/layer_normalization_1/beta7transformer_decoder_3/multi_head_attention/query/kernel5transformer_decoder_3/multi_head_attention/query/bias5transformer_decoder_3/multi_head_attention/key/kernel3transformer_decoder_3/multi_head_attention/key/bias7transformer_decoder_3/multi_head_attention/value/kernel5transformer_decoder_3/multi_head_attention/value/biasBtransformer_decoder_3/multi_head_attention/attention_output/kernel@transformer_decoder_3/multi_head_attention/attention_output/bias/transformer_decoder_3/layer_normalization/gamma.transformer_decoder_3/layer_normalization/beta"transformer_decoder_3/dense/kernel transformer_decoder_3/dense/bias$transformer_decoder_3/dense_1/kernel"transformer_decoder_3/dense_1/bias1transformer_decoder_3/layer_normalization_1/gamma0transformer_decoder_3/layer_normalization_1/betadense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *I
_read_only_resource_inputs+
)'	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_334728
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?G
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*embedding_3/embeddings/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpNtoken_and_position_embedding_2/token_embedding3/embeddings/Read/ReadVariableOpQtoken_and_position_embedding_2/position_embedding3/embeddings/Read/ReadVariableOpKtransformer_encoder_3/multi_head_attention/query/kernel/Read/ReadVariableOpItransformer_encoder_3/multi_head_attention/query/bias/Read/ReadVariableOpItransformer_encoder_3/multi_head_attention/key/kernel/Read/ReadVariableOpGtransformer_encoder_3/multi_head_attention/key/bias/Read/ReadVariableOpKtransformer_encoder_3/multi_head_attention/value/kernel/Read/ReadVariableOpItransformer_encoder_3/multi_head_attention/value/bias/Read/ReadVariableOpVtransformer_encoder_3/multi_head_attention/attention_output/kernel/Read/ReadVariableOpTtransformer_encoder_3/multi_head_attention/attention_output/bias/Read/ReadVariableOpCtransformer_encoder_3/layer_normalization/gamma/Read/ReadVariableOpBtransformer_encoder_3/layer_normalization/beta/Read/ReadVariableOpEtransformer_encoder_3/layer_normalization_1/gamma/Read/ReadVariableOpDtransformer_encoder_3/layer_normalization_1/beta/Read/ReadVariableOp6transformer_encoder_3/dense/kernel/Read/ReadVariableOp4transformer_encoder_3/dense/bias/Read/ReadVariableOp8transformer_encoder_3/dense_1/kernel/Read/ReadVariableOp6transformer_encoder_3/dense_1/bias/Read/ReadVariableOpKtransformer_decoder_3/multi_head_attention/query/kernel/Read/ReadVariableOpItransformer_decoder_3/multi_head_attention/query/bias/Read/ReadVariableOpItransformer_decoder_3/multi_head_attention/key/kernel/Read/ReadVariableOpGtransformer_decoder_3/multi_head_attention/key/bias/Read/ReadVariableOpKtransformer_decoder_3/multi_head_attention/value/kernel/Read/ReadVariableOpItransformer_decoder_3/multi_head_attention/value/bias/Read/ReadVariableOpVtransformer_decoder_3/multi_head_attention/attention_output/kernel/Read/ReadVariableOpTtransformer_decoder_3/multi_head_attention/attention_output/bias/Read/ReadVariableOpCtransformer_decoder_3/layer_normalization/gamma/Read/ReadVariableOpBtransformer_decoder_3/layer_normalization/beta/Read/ReadVariableOpEtransformer_decoder_3/layer_normalization_1/gamma/Read/ReadVariableOpDtransformer_decoder_3/layer_normalization_1/beta/Read/ReadVariableOp6transformer_decoder_3/dense/kernel/Read/ReadVariableOp4transformer_decoder_3/dense/bias/Read/ReadVariableOp8transformer_decoder_3/dense_1/kernel/Read/ReadVariableOp6transformer_decoder_3/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp1Adam/embedding_3/embeddings/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOpUAdam/token_and_position_embedding_2/token_embedding3/embeddings/m/Read/ReadVariableOpXAdam/token_and_position_embedding_2/position_embedding3/embeddings/m/Read/ReadVariableOpRAdam/transformer_encoder_3/multi_head_attention/query/kernel/m/Read/ReadVariableOpPAdam/transformer_encoder_3/multi_head_attention/query/bias/m/Read/ReadVariableOpPAdam/transformer_encoder_3/multi_head_attention/key/kernel/m/Read/ReadVariableOpNAdam/transformer_encoder_3/multi_head_attention/key/bias/m/Read/ReadVariableOpRAdam/transformer_encoder_3/multi_head_attention/value/kernel/m/Read/ReadVariableOpPAdam/transformer_encoder_3/multi_head_attention/value/bias/m/Read/ReadVariableOp]Adam/transformer_encoder_3/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOp[Adam/transformer_encoder_3/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpJAdam/transformer_encoder_3/layer_normalization/gamma/m/Read/ReadVariableOpIAdam/transformer_encoder_3/layer_normalization/beta/m/Read/ReadVariableOpLAdam/transformer_encoder_3/layer_normalization_1/gamma/m/Read/ReadVariableOpKAdam/transformer_encoder_3/layer_normalization_1/beta/m/Read/ReadVariableOp=Adam/transformer_encoder_3/dense/kernel/m/Read/ReadVariableOp;Adam/transformer_encoder_3/dense/bias/m/Read/ReadVariableOp?Adam/transformer_encoder_3/dense_1/kernel/m/Read/ReadVariableOp=Adam/transformer_encoder_3/dense_1/bias/m/Read/ReadVariableOpRAdam/transformer_decoder_3/multi_head_attention/query/kernel/m/Read/ReadVariableOpPAdam/transformer_decoder_3/multi_head_attention/query/bias/m/Read/ReadVariableOpPAdam/transformer_decoder_3/multi_head_attention/key/kernel/m/Read/ReadVariableOpNAdam/transformer_decoder_3/multi_head_attention/key/bias/m/Read/ReadVariableOpRAdam/transformer_decoder_3/multi_head_attention/value/kernel/m/Read/ReadVariableOpPAdam/transformer_decoder_3/multi_head_attention/value/bias/m/Read/ReadVariableOp]Adam/transformer_decoder_3/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOp[Adam/transformer_decoder_3/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpJAdam/transformer_decoder_3/layer_normalization/gamma/m/Read/ReadVariableOpIAdam/transformer_decoder_3/layer_normalization/beta/m/Read/ReadVariableOpLAdam/transformer_decoder_3/layer_normalization_1/gamma/m/Read/ReadVariableOpKAdam/transformer_decoder_3/layer_normalization_1/beta/m/Read/ReadVariableOp=Adam/transformer_decoder_3/dense/kernel/m/Read/ReadVariableOp;Adam/transformer_decoder_3/dense/bias/m/Read/ReadVariableOp?Adam/transformer_decoder_3/dense_1/kernel/m/Read/ReadVariableOp=Adam/transformer_decoder_3/dense_1/bias/m/Read/ReadVariableOp1Adam/embedding_3/embeddings/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpUAdam/token_and_position_embedding_2/token_embedding3/embeddings/v/Read/ReadVariableOpXAdam/token_and_position_embedding_2/position_embedding3/embeddings/v/Read/ReadVariableOpRAdam/transformer_encoder_3/multi_head_attention/query/kernel/v/Read/ReadVariableOpPAdam/transformer_encoder_3/multi_head_attention/query/bias/v/Read/ReadVariableOpPAdam/transformer_encoder_3/multi_head_attention/key/kernel/v/Read/ReadVariableOpNAdam/transformer_encoder_3/multi_head_attention/key/bias/v/Read/ReadVariableOpRAdam/transformer_encoder_3/multi_head_attention/value/kernel/v/Read/ReadVariableOpPAdam/transformer_encoder_3/multi_head_attention/value/bias/v/Read/ReadVariableOp]Adam/transformer_encoder_3/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOp[Adam/transformer_encoder_3/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpJAdam/transformer_encoder_3/layer_normalization/gamma/v/Read/ReadVariableOpIAdam/transformer_encoder_3/layer_normalization/beta/v/Read/ReadVariableOpLAdam/transformer_encoder_3/layer_normalization_1/gamma/v/Read/ReadVariableOpKAdam/transformer_encoder_3/layer_normalization_1/beta/v/Read/ReadVariableOp=Adam/transformer_encoder_3/dense/kernel/v/Read/ReadVariableOp;Adam/transformer_encoder_3/dense/bias/v/Read/ReadVariableOp?Adam/transformer_encoder_3/dense_1/kernel/v/Read/ReadVariableOp=Adam/transformer_encoder_3/dense_1/bias/v/Read/ReadVariableOpRAdam/transformer_decoder_3/multi_head_attention/query/kernel/v/Read/ReadVariableOpPAdam/transformer_decoder_3/multi_head_attention/query/bias/v/Read/ReadVariableOpPAdam/transformer_decoder_3/multi_head_attention/key/kernel/v/Read/ReadVariableOpNAdam/transformer_decoder_3/multi_head_attention/key/bias/v/Read/ReadVariableOpRAdam/transformer_decoder_3/multi_head_attention/value/kernel/v/Read/ReadVariableOpPAdam/transformer_decoder_3/multi_head_attention/value/bias/v/Read/ReadVariableOp]Adam/transformer_decoder_3/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOp[Adam/transformer_decoder_3/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpJAdam/transformer_decoder_3/layer_normalization/gamma/v/Read/ReadVariableOpIAdam/transformer_decoder_3/layer_normalization/beta/v/Read/ReadVariableOpLAdam/transformer_decoder_3/layer_normalization_1/gamma/v/Read/ReadVariableOpKAdam/transformer_decoder_3/layer_normalization_1/beta/v/Read/ReadVariableOp=Adam/transformer_decoder_3/dense/kernel/v/Read/ReadVariableOp;Adam/transformer_decoder_3/dense/bias/v/Read/ReadVariableOp?Adam/transformer_decoder_3/dense_1/kernel/v/Read/ReadVariableOp=Adam/transformer_decoder_3/dense_1/bias/v/Read/ReadVariableOpConst_6*?
Tin?
?2?		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_337127
?3
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding_3/embeddingsdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias:token_and_position_embedding_2/token_embedding3/embeddings=token_and_position_embedding_2/position_embedding3/embeddings7transformer_encoder_3/multi_head_attention/query/kernel5transformer_encoder_3/multi_head_attention/query/bias5transformer_encoder_3/multi_head_attention/key/kernel3transformer_encoder_3/multi_head_attention/key/bias7transformer_encoder_3/multi_head_attention/value/kernel5transformer_encoder_3/multi_head_attention/value/biasBtransformer_encoder_3/multi_head_attention/attention_output/kernel@transformer_encoder_3/multi_head_attention/attention_output/bias/transformer_encoder_3/layer_normalization/gamma.transformer_encoder_3/layer_normalization/beta1transformer_encoder_3/layer_normalization_1/gamma0transformer_encoder_3/layer_normalization_1/beta"transformer_encoder_3/dense/kernel transformer_encoder_3/dense/bias$transformer_encoder_3/dense_1/kernel"transformer_encoder_3/dense_1/bias7transformer_decoder_3/multi_head_attention/query/kernel5transformer_decoder_3/multi_head_attention/query/bias5transformer_decoder_3/multi_head_attention/key/kernel3transformer_decoder_3/multi_head_attention/key/bias7transformer_decoder_3/multi_head_attention/value/kernel5transformer_decoder_3/multi_head_attention/value/biasBtransformer_decoder_3/multi_head_attention/attention_output/kernel@transformer_decoder_3/multi_head_attention/attention_output/bias/transformer_decoder_3/layer_normalization/gamma.transformer_decoder_3/layer_normalization/beta1transformer_decoder_3/layer_normalization_1/gamma0transformer_decoder_3/layer_normalization_1/beta"transformer_decoder_3/dense/kernel transformer_decoder_3/dense/bias$transformer_decoder_3/dense_1/kernel"transformer_decoder_3/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotal_1count_1totalcountAdam/embedding_3/embeddings/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAAdam/token_and_position_embedding_2/token_embedding3/embeddings/mDAdam/token_and_position_embedding_2/position_embedding3/embeddings/m>Adam/transformer_encoder_3/multi_head_attention/query/kernel/m<Adam/transformer_encoder_3/multi_head_attention/query/bias/m<Adam/transformer_encoder_3/multi_head_attention/key/kernel/m:Adam/transformer_encoder_3/multi_head_attention/key/bias/m>Adam/transformer_encoder_3/multi_head_attention/value/kernel/m<Adam/transformer_encoder_3/multi_head_attention/value/bias/mIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/mGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/m6Adam/transformer_encoder_3/layer_normalization/gamma/m5Adam/transformer_encoder_3/layer_normalization/beta/m8Adam/transformer_encoder_3/layer_normalization_1/gamma/m7Adam/transformer_encoder_3/layer_normalization_1/beta/m)Adam/transformer_encoder_3/dense/kernel/m'Adam/transformer_encoder_3/dense/bias/m+Adam/transformer_encoder_3/dense_1/kernel/m)Adam/transformer_encoder_3/dense_1/bias/m>Adam/transformer_decoder_3/multi_head_attention/query/kernel/m<Adam/transformer_decoder_3/multi_head_attention/query/bias/m<Adam/transformer_decoder_3/multi_head_attention/key/kernel/m:Adam/transformer_decoder_3/multi_head_attention/key/bias/m>Adam/transformer_decoder_3/multi_head_attention/value/kernel/m<Adam/transformer_decoder_3/multi_head_attention/value/bias/mIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/mGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/m6Adam/transformer_decoder_3/layer_normalization/gamma/m5Adam/transformer_decoder_3/layer_normalization/beta/m8Adam/transformer_decoder_3/layer_normalization_1/gamma/m7Adam/transformer_decoder_3/layer_normalization_1/beta/m)Adam/transformer_decoder_3/dense/kernel/m'Adam/transformer_decoder_3/dense/bias/m+Adam/transformer_decoder_3/dense_1/kernel/m)Adam/transformer_decoder_3/dense_1/bias/mAdam/embedding_3/embeddings/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAAdam/token_and_position_embedding_2/token_embedding3/embeddings/vDAdam/token_and_position_embedding_2/position_embedding3/embeddings/v>Adam/transformer_encoder_3/multi_head_attention/query/kernel/v<Adam/transformer_encoder_3/multi_head_attention/query/bias/v<Adam/transformer_encoder_3/multi_head_attention/key/kernel/v:Adam/transformer_encoder_3/multi_head_attention/key/bias/v>Adam/transformer_encoder_3/multi_head_attention/value/kernel/v<Adam/transformer_encoder_3/multi_head_attention/value/bias/vIAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/vGAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/v6Adam/transformer_encoder_3/layer_normalization/gamma/v5Adam/transformer_encoder_3/layer_normalization/beta/v8Adam/transformer_encoder_3/layer_normalization_1/gamma/v7Adam/transformer_encoder_3/layer_normalization_1/beta/v)Adam/transformer_encoder_3/dense/kernel/v'Adam/transformer_encoder_3/dense/bias/v+Adam/transformer_encoder_3/dense_1/kernel/v)Adam/transformer_encoder_3/dense_1/bias/v>Adam/transformer_decoder_3/multi_head_attention/query/kernel/v<Adam/transformer_decoder_3/multi_head_attention/query/bias/v<Adam/transformer_decoder_3/multi_head_attention/key/kernel/v:Adam/transformer_decoder_3/multi_head_attention/key/bias/v>Adam/transformer_decoder_3/multi_head_attention/value/kernel/v<Adam/transformer_decoder_3/multi_head_attention/value/bias/vIAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/vGAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/v6Adam/transformer_decoder_3/layer_normalization/gamma/v5Adam/transformer_decoder_3/layer_normalization/beta/v8Adam/transformer_decoder_3/layer_normalization_1/gamma/v7Adam/transformer_decoder_3/layer_normalization_1/beta/v)Adam/transformer_decoder_3/dense/kernel/v'Adam/transformer_decoder_3/dense/bias/v+Adam/transformer_decoder_3/dense_1/kernel/v)Adam/transformer_decoder_3/dense_1/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_337518??,
?	
?
G__inference_embedding_3_layer_call_and_return_conditional_losses_335785

inputs)
embedding_lookup_335779:
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
embedding_lookupResourceGatherembedding_lookup_335779Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/335779*+
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/335779*+
_output_shapes
:????????? ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:????????? : 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?

(__inference_model_2_layer_call_fn_334354

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24:

unknown_25: 

unknown_26:

unknown_27: 

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:@

unknown_39:@

unknown_40:@ 

unknown_41: 
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
unknown_41*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *I
_read_only_resource_inputs+
)'	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_334173o
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
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namePhrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
Token_role:
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
?7
!__inference__wrapped_model_332788

phrase

token_role]
Ymodel_2_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle^
Zmodel_2_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	4
0model_2_text_vectorization_string_lookup_equal_y7
3model_2_text_vectorization_string_lookup_selectv2_t	b
Omodel_2_token_and_position_embedding_2_token_embedding3_embedding_lookup_332449:	?d
Rmodel_2_token_and_position_embedding_2_position_embedding3_readvariableop_resource: =
+model_2_embedding_3_embedding_lookup_332473:t
^model_2_transformer_encoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:f
Tmodel_2_transformer_encoder_3_multi_head_attention_query_add_readvariableop_resource:r
\model_2_transformer_encoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:d
Rmodel_2_transformer_encoder_3_multi_head_attention_key_add_readvariableop_resource:t
^model_2_transformer_encoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:f
Tmodel_2_transformer_encoder_3_multi_head_attention_value_add_readvariableop_resource:
imodel_2_transformer_encoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:m
_model_2_transformer_encoder_3_multi_head_attention_attention_output_add_readvariableop_resource:e
Wmodel_2_transformer_encoder_3_layer_normalization_batchnorm_mul_readvariableop_resource:a
Smodel_2_transformer_encoder_3_layer_normalization_batchnorm_readvariableop_resource:W
Emodel_2_transformer_encoder_3_dense_tensordot_readvariableop_resource:Q
Cmodel_2_transformer_encoder_3_dense_biasadd_readvariableop_resource:Y
Gmodel_2_transformer_encoder_3_dense_1_tensordot_readvariableop_resource:S
Emodel_2_transformer_encoder_3_dense_1_biasadd_readvariableop_resource:g
Ymodel_2_transformer_encoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:c
Umodel_2_transformer_encoder_3_layer_normalization_1_batchnorm_readvariableop_resource:t
^model_2_transformer_decoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:f
Tmodel_2_transformer_decoder_3_multi_head_attention_query_add_readvariableop_resource:r
\model_2_transformer_decoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:d
Rmodel_2_transformer_decoder_3_multi_head_attention_key_add_readvariableop_resource:t
^model_2_transformer_decoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:f
Tmodel_2_transformer_decoder_3_multi_head_attention_value_add_readvariableop_resource:
imodel_2_transformer_decoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:m
_model_2_transformer_decoder_3_multi_head_attention_attention_output_add_readvariableop_resource:e
Wmodel_2_transformer_decoder_3_layer_normalization_batchnorm_mul_readvariableop_resource:a
Smodel_2_transformer_decoder_3_layer_normalization_batchnorm_readvariableop_resource:W
Emodel_2_transformer_decoder_3_dense_tensordot_readvariableop_resource:Q
Cmodel_2_transformer_decoder_3_dense_biasadd_readvariableop_resource:Y
Gmodel_2_transformer_decoder_3_dense_1_tensordot_readvariableop_resource:S
Emodel_2_transformer_decoder_3_dense_1_biasadd_readvariableop_resource:g
Ymodel_2_transformer_decoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:c
Umodel_2_transformer_decoder_3_layer_normalization_1_batchnorm_readvariableop_resource:@
.model_2_dense_4_matmul_readvariableop_resource:@=
/model_2_dense_4_biasadd_readvariableop_resource:@@
.model_2_dense_5_matmul_readvariableop_resource:@ =
/model_2_dense_5_biasadd_readvariableop_resource: 
identity??&model_2/dense_4/BiasAdd/ReadVariableOp?%model_2/dense_4/MatMul/ReadVariableOp?&model_2/dense_5/BiasAdd/ReadVariableOp?%model_2/dense_5/MatMul/ReadVariableOp?$model_2/embedding_3/embedding_lookup?Lmodel_2/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?Imodel_2/token_and_position_embedding_2/position_embedding3/ReadVariableOp?Hmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookup?:model_2/transformer_decoder_3/dense/BiasAdd/ReadVariableOp?<model_2/transformer_decoder_3/dense/Tensordot/ReadVariableOp?<model_2/transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp?>model_2/transformer_decoder_3/dense_1/Tensordot/ReadVariableOp?Jmodel_2/transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp?Nmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp?Lmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp?Pmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp?Vmodel_2/transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp?`model_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Imodel_2/transformer_decoder_3/multi_head_attention/key/add/ReadVariableOp?Smodel_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Kmodel_2/transformer_decoder_3/multi_head_attention/query/add/ReadVariableOp?Umodel_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Kmodel_2/transformer_decoder_3/multi_head_attention/value/add/ReadVariableOp?Umodel_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp?:model_2/transformer_encoder_3/dense/BiasAdd/ReadVariableOp?<model_2/transformer_encoder_3/dense/Tensordot/ReadVariableOp?<model_2/transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp?>model_2/transformer_encoder_3/dense_1/Tensordot/ReadVariableOp?Jmodel_2/transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp?Nmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp?Lmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp?Pmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp?Vmodel_2/transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp?`model_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Imodel_2/transformer_encoder_3/multi_head_attention/key/add/ReadVariableOp?Smodel_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Kmodel_2/transformer_encoder_3/multi_head_attention/query/add/ReadVariableOp?Umodel_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Kmodel_2/transformer_encoder_3/multi_head_attention/value/add/ReadVariableOp?Umodel_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp?
"model_2/text_vectorization/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????m
,model_2/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
4model_2/text_vectorization/StringSplit/StringSplitV2StringSplitV2+model_2/text_vectorization/Squeeze:output:05model_2/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
:model_2/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
<model_2/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
<model_2/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
4model_2/text_vectorization/StringSplit/strided_sliceStridedSlice>model_2/text_vectorization/StringSplit/StringSplitV2:indices:0Cmodel_2/text_vectorization/StringSplit/strided_slice/stack:output:0Emodel_2/text_vectorization/StringSplit/strided_slice/stack_1:output:0Emodel_2/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
<model_2/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>model_2/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>model_2/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6model_2/text_vectorization/StringSplit/strided_slice_1StridedSlice<model_2/text_vectorization/StringSplit/StringSplitV2:shape:0Emodel_2/text_vectorization/StringSplit/strided_slice_1/stack:output:0Gmodel_2/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Gmodel_2/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
]model_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast=model_2/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
_model_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast?model_2/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
gmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeamodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
gmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
fmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdpmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0pmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
kmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateromodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0tmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
fmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastmmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
emodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxamodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0rmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
gmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
emodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2nmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0pmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
emodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuljmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumcmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumcmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0mmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
imodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
jmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountamodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0mmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0rmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
dmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
_model_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumqmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0mmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
hmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
dmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
_model_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2qmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0emodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0mmodel_2/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Lmodel_2/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Ymodel_2_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle=model_2/text_vectorization/StringSplit/StringSplitV2:values:0Zmodel_2_text_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
.model_2/text_vectorization/string_lookup/EqualEqual=model_2/text_vectorization/StringSplit/StringSplitV2:values:00model_2_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
1model_2/text_vectorization/string_lookup/SelectV2SelectV22model_2/text_vectorization/string_lookup/Equal:z:03model_2_text_vectorization_string_lookup_selectv2_tUmodel_2/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
1model_2/text_vectorization/string_lookup/IdentityIdentity:model_2/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????y
7model_2/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
/model_2/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
>model_2/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor8model_2/text_vectorization/RaggedToTensor/Const:output:0:model_2/text_vectorization/string_lookup/Identity:output:0@model_2/text_vectorization/RaggedToTensor/default_value:output:0?model_2/text_vectorization/StringSplit/strided_slice_1:output:0=model_2/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
Hmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookupResourceGatherOmodel_2_token_and_position_embedding_2_token_embedding3_embedding_lookup_332449Gmodel_2/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*b
_classX
VTloc:@model_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/332449*+
_output_shapes
:????????? *
dtype0?
Qmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/IdentityIdentityQmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookup:output:0*
T0*b
_classX
VTloc:@model_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/332449*+
_output_shapes
:????????? ?
Smodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1IdentityZmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
@model_2/token_and_position_embedding_2/position_embedding3/ShapeShape\model_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Nmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Pmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Pmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hmodel_2/token_and_position_embedding_2/position_embedding3/strided_sliceStridedSliceImodel_2/token_and_position_embedding_2/position_embedding3/Shape:output:0Wmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice/stack:output:0Ymodel_2/token_and_position_embedding_2/position_embedding3/strided_slice/stack_1:output:0Ymodel_2/token_and_position_embedding_2/position_embedding3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Imodel_2/token_and_position_embedding_2/position_embedding3/ReadVariableOpReadVariableOpRmodel_2_token_and_position_embedding_2_position_embedding3_readvariableop_resource*
_output_shapes

: *
dtype0?
@model_2/token_and_position_embedding_2/position_embedding3/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Bmodel_2/token_and_position_embedding_2/position_embedding3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Rmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Pmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stackPackImodel_2/token_and_position_embedding_2/position_embedding3/Const:output:0[model_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Tmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Rmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1PackQmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice:output:0]model_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Tmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Rmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2PackKmodel_2/token_and_position_embedding_2/position_embedding3/Const_1:output:0]model_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Jmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1StridedSliceQmodel_2/token_and_position_embedding_2/position_embedding3/ReadVariableOp:value:0Ymodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack:output:0[model_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1:output:0[model_2/token_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
Fmodel_2/token_and_position_embedding_2/position_embedding3/BroadcastToBroadcastToSmodel_2/token_and_position_embedding_2/position_embedding3/strided_slice_1:output:0Imodel_2/token_and_position_embedding_2/position_embedding3/Shape:output:0*
T0*+
_output_shapes
:????????? ?
*model_2/token_and_position_embedding_2/addAddV2\model_2/token_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1:output:0Omodel_2/token_and_position_embedding_2/position_embedding3/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? m
model_2/embedding_3/CastCast
token_role*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
$model_2/embedding_3/embedding_lookupResourceGather+model_2_embedding_3_embedding_lookup_332473model_2/embedding_3/Cast:y:0*
Tindices0*>
_class4
20loc:@model_2/embedding_3/embedding_lookup/332473*+
_output_shapes
:????????? *
dtype0?
-model_2/embedding_3/embedding_lookup/IdentityIdentity-model_2/embedding_3/embedding_lookup:output:0*
T0*>
_class4
20loc:@model_2/embedding_3/embedding_lookup/332473*+
_output_shapes
:????????? ?
/model_2/embedding_3/embedding_lookup/Identity_1Identity6model_2/embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
model_2/add_2/addAddV2.model_2/token_and_position_embedding_2/add:z:08model_2/embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:????????? ?
Umodel_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_encoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fmodel_2/transformer_encoder_3/multi_head_attention/query/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Kmodel_2/transformer_encoder_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_encoder_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
<model_2/transformer_encoder_3/multi_head_attention/query/addAddV2Omodel_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum:output:0Smodel_2/transformer_encoder_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Smodel_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_encoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Dmodel_2/transformer_encoder_3/multi_head_attention/key/einsum/EinsumEinsummodel_2/add_2/add:z:0[model_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Imodel_2/transformer_encoder_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_encoder_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
:model_2/transformer_encoder_3/multi_head_attention/key/addAddV2Mmodel_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum:output:0Qmodel_2/transformer_encoder_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Umodel_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_encoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fmodel_2/transformer_encoder_3/multi_head_attention/value/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Kmodel_2/transformer_encoder_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_encoder_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
<model_2/transformer_encoder_3/multi_head_attention/value/addAddV2Omodel_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum:output:0Smodel_2/transformer_encoder_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? }
8model_2/transformer_encoder_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
6model_2/transformer_encoder_3/multi_head_attention/MulMul@model_2/transformer_encoder_3/multi_head_attention/query/add:z:0Amodel_2/transformer_encoder_3/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
@model_2/transformer_encoder_3/multi_head_attention/einsum/EinsumEinsum>model_2/transformer_encoder_3/multi_head_attention/key/add:z:0:model_2/transformer_encoder_3/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
Bmodel_2/transformer_encoder_3/multi_head_attention/softmax/SoftmaxSoftmaxImodel_2/transformer_encoder_3/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
Emodel_2/transformer_encoder_3/multi_head_attention/dropout_2/IdentityIdentityLmodel_2/transformer_encoder_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
Bmodel_2/transformer_encoder_3/multi_head_attention/einsum_1/EinsumEinsumNmodel_2/transformer_encoder_3/multi_head_attention/dropout_2/Identity:output:0@model_2/transformer_encoder_3/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
`model_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_encoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Qmodel_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/EinsumEinsumKmodel_2/transformer_encoder_3/multi_head_attention/einsum_1/Einsum:output:0hmodel_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Vmodel_2/transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_encoder_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
Gmodel_2/transformer_encoder_3/multi_head_attention/attention_output/addAddV2Zmodel_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum:output:0^model_2/transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
.model_2/transformer_encoder_3/dropout/IdentityIdentityKmodel_2/transformer_encoder_3/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
!model_2/transformer_encoder_3/addAddV2model_2/add_2/add:z:07model_2/transformer_encoder_3/dropout/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Pmodel_2/transformer_encoder_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
>model_2/transformer_encoder_3/layer_normalization/moments/meanMean%model_2/transformer_encoder_3/add:z:0Ymodel_2/transformer_encoder_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Fmodel_2/transformer_encoder_3/layer_normalization/moments/StopGradientStopGradientGmodel_2/transformer_encoder_3/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Kmodel_2/transformer_encoder_3/layer_normalization/moments/SquaredDifferenceSquaredDifference%model_2/transformer_encoder_3/add:z:0Omodel_2/transformer_encoder_3/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Tmodel_2/transformer_encoder_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_2/transformer_encoder_3/layer_normalization/moments/varianceMeanOmodel_2/transformer_encoder_3/layer_normalization/moments/SquaredDifference:z:0]model_2/transformer_encoder_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Amodel_2/transformer_encoder_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
?model_2/transformer_encoder_3/layer_normalization/batchnorm/addAddV2Kmodel_2/transformer_encoder_3/layer_normalization/moments/variance:output:0Jmodel_2/transformer_encoder_3/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_encoder_3/layer_normalization/batchnorm/RsqrtRsqrtCmodel_2/transformer_encoder_3/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Nmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_encoder_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_2/transformer_encoder_3/layer_normalization/batchnorm/mulMulEmodel_2/transformer_encoder_3/layer_normalization/batchnorm/Rsqrt:y:0Vmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul_1Mul%model_2/transformer_encoder_3/add:z:0Cmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul_2MulGmodel_2/transformer_encoder_3/layer_normalization/moments/mean:output:0Cmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Jmodel_2/transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_encoder_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_2/transformer_encoder_3/layer_normalization/batchnorm/subSubRmodel_2/transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_encoder_3/layer_normalization/batchnorm/add_1AddV2Emodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul_1:z:0Cmodel_2/transformer_encoder_3/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
<model_2/transformer_encoder_3/dense/Tensordot/ReadVariableOpReadVariableOpEmodel_2_transformer_encoder_3_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0|
2model_2/transformer_encoder_3/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
2model_2/transformer_encoder_3/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
3model_2/transformer_encoder_3/dense/Tensordot/ShapeShapeEmodel_2/transformer_encoder_3/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:}
;model_2/transformer_encoder_3/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_2/transformer_encoder_3/dense/Tensordot/GatherV2GatherV2<model_2/transformer_encoder_3/dense/Tensordot/Shape:output:0;model_2/transformer_encoder_3/dense/Tensordot/free:output:0Dmodel_2/transformer_encoder_3/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=model_2/transformer_encoder_3/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_2/transformer_encoder_3/dense/Tensordot/GatherV2_1GatherV2<model_2/transformer_encoder_3/dense/Tensordot/Shape:output:0;model_2/transformer_encoder_3/dense/Tensordot/axes:output:0Fmodel_2/transformer_encoder_3/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
3model_2/transformer_encoder_3/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2model_2/transformer_encoder_3/dense/Tensordot/ProdProd?model_2/transformer_encoder_3/dense/Tensordot/GatherV2:output:0<model_2/transformer_encoder_3/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
5model_2/transformer_encoder_3/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
4model_2/transformer_encoder_3/dense/Tensordot/Prod_1ProdAmodel_2/transformer_encoder_3/dense/Tensordot/GatherV2_1:output:0>model_2/transformer_encoder_3/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: {
9model_2/transformer_encoder_3/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4model_2/transformer_encoder_3/dense/Tensordot/concatConcatV2;model_2/transformer_encoder_3/dense/Tensordot/free:output:0;model_2/transformer_encoder_3/dense/Tensordot/axes:output:0Bmodel_2/transformer_encoder_3/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
3model_2/transformer_encoder_3/dense/Tensordot/stackPack;model_2/transformer_encoder_3/dense/Tensordot/Prod:output:0=model_2/transformer_encoder_3/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
7model_2/transformer_encoder_3/dense/Tensordot/transpose	TransposeEmodel_2/transformer_encoder_3/layer_normalization/batchnorm/add_1:z:0=model_2/transformer_encoder_3/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
5model_2/transformer_encoder_3/dense/Tensordot/ReshapeReshape;model_2/transformer_encoder_3/dense/Tensordot/transpose:y:0<model_2/transformer_encoder_3/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
4model_2/transformer_encoder_3/dense/Tensordot/MatMulMatMul>model_2/transformer_encoder_3/dense/Tensordot/Reshape:output:0Dmodel_2/transformer_encoder_3/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
5model_2/transformer_encoder_3/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:}
;model_2/transformer_encoder_3/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_2/transformer_encoder_3/dense/Tensordot/concat_1ConcatV2?model_2/transformer_encoder_3/dense/Tensordot/GatherV2:output:0>model_2/transformer_encoder_3/dense/Tensordot/Const_2:output:0Dmodel_2/transformer_encoder_3/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
-model_2/transformer_encoder_3/dense/TensordotReshape>model_2/transformer_encoder_3/dense/Tensordot/MatMul:product:0?model_2/transformer_encoder_3/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
:model_2/transformer_encoder_3/dense/BiasAdd/ReadVariableOpReadVariableOpCmodel_2_transformer_encoder_3_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
+model_2/transformer_encoder_3/dense/BiasAddBiasAdd6model_2/transformer_encoder_3/dense/Tensordot:output:0Bmodel_2/transformer_encoder_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
(model_2/transformer_encoder_3/dense/ReluRelu4model_2/transformer_encoder_3/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
>model_2/transformer_encoder_3/dense_1/Tensordot/ReadVariableOpReadVariableOpGmodel_2_transformer_encoder_3_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0~
4model_2/transformer_encoder_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
4model_2/transformer_encoder_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
5model_2/transformer_encoder_3/dense_1/Tensordot/ShapeShape6model_2/transformer_encoder_3/dense/Relu:activations:0*
T0*
_output_shapes
:
=model_2/transformer_encoder_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_2/transformer_encoder_3/dense_1/Tensordot/GatherV2GatherV2>model_2/transformer_encoder_3/dense_1/Tensordot/Shape:output:0=model_2/transformer_encoder_3/dense_1/Tensordot/free:output:0Fmodel_2/transformer_encoder_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
?model_2/transformer_encoder_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_2/transformer_encoder_3/dense_1/Tensordot/GatherV2_1GatherV2>model_2/transformer_encoder_3/dense_1/Tensordot/Shape:output:0=model_2/transformer_encoder_3/dense_1/Tensordot/axes:output:0Hmodel_2/transformer_encoder_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5model_2/transformer_encoder_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4model_2/transformer_encoder_3/dense_1/Tensordot/ProdProdAmodel_2/transformer_encoder_3/dense_1/Tensordot/GatherV2:output:0>model_2/transformer_encoder_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
7model_2/transformer_encoder_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
6model_2/transformer_encoder_3/dense_1/Tensordot/Prod_1ProdCmodel_2/transformer_encoder_3/dense_1/Tensordot/GatherV2_1:output:0@model_2/transformer_encoder_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: }
;model_2/transformer_encoder_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_2/transformer_encoder_3/dense_1/Tensordot/concatConcatV2=model_2/transformer_encoder_3/dense_1/Tensordot/free:output:0=model_2/transformer_encoder_3/dense_1/Tensordot/axes:output:0Dmodel_2/transformer_encoder_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5model_2/transformer_encoder_3/dense_1/Tensordot/stackPack=model_2/transformer_encoder_3/dense_1/Tensordot/Prod:output:0?model_2/transformer_encoder_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
9model_2/transformer_encoder_3/dense_1/Tensordot/transpose	Transpose6model_2/transformer_encoder_3/dense/Relu:activations:0?model_2/transformer_encoder_3/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
7model_2/transformer_encoder_3/dense_1/Tensordot/ReshapeReshape=model_2/transformer_encoder_3/dense_1/Tensordot/transpose:y:0>model_2/transformer_encoder_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
6model_2/transformer_encoder_3/dense_1/Tensordot/MatMulMatMul@model_2/transformer_encoder_3/dense_1/Tensordot/Reshape:output:0Fmodel_2/transformer_encoder_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
7model_2/transformer_encoder_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
=model_2/transformer_encoder_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_2/transformer_encoder_3/dense_1/Tensordot/concat_1ConcatV2Amodel_2/transformer_encoder_3/dense_1/Tensordot/GatherV2:output:0@model_2/transformer_encoder_3/dense_1/Tensordot/Const_2:output:0Fmodel_2/transformer_encoder_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
/model_2/transformer_encoder_3/dense_1/TensordotReshape@model_2/transformer_encoder_3/dense_1/Tensordot/MatMul:product:0Amodel_2/transformer_encoder_3/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
<model_2/transformer_encoder_3/dense_1/BiasAdd/ReadVariableOpReadVariableOpEmodel_2_transformer_encoder_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-model_2/transformer_encoder_3/dense_1/BiasAddBiasAdd8model_2/transformer_encoder_3/dense_1/Tensordot:output:0Dmodel_2/transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
0model_2/transformer_encoder_3/dropout_1/IdentityIdentity6model_2/transformer_encoder_3/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
#model_2/transformer_encoder_3/add_1AddV2Emodel_2/transformer_encoder_3/layer_normalization/batchnorm/add_1:z:09model_2/transformer_encoder_3/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Rmodel_2/transformer_encoder_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_2/transformer_encoder_3/layer_normalization_1/moments/meanMean'model_2/transformer_encoder_3/add_1:z:0[model_2/transformer_encoder_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Hmodel_2/transformer_encoder_3/layer_normalization_1/moments/StopGradientStopGradientImodel_2/transformer_encoder_3/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Mmodel_2/transformer_encoder_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifference'model_2/transformer_encoder_3/add_1:z:0Qmodel_2/transformer_encoder_3/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Vmodel_2/transformer_encoder_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_2/transformer_encoder_3/layer_normalization_1/moments/varianceMeanQmodel_2/transformer_encoder_3/layer_normalization_1/moments/SquaredDifference:z:0_model_2/transformer_encoder_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Cmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Amodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/addAddV2Mmodel_2/transformer_encoder_3/layer_normalization_1/moments/variance:output:0Lmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/RsqrtRsqrtEmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Pmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_2_transformer_encoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mulMulGmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/Rsqrt:y:0Xmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul_1Mul'model_2/transformer_encoder_3/add_1:z:0Emodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul_2MulImodel_2/transformer_encoder_3/layer_normalization_1/moments/mean:output:0Emodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Lmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpUmodel_2_transformer_encoder_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/subSubTmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0Gmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add_1AddV2Gmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul_1:z:0Emodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
#model_2/transformer_decoder_3/ShapeShapeGmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:{
1model_2/transformer_decoder_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_2/transformer_decoder_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_2/transformer_decoder_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_2/transformer_decoder_3/strided_sliceStridedSlice,model_2/transformer_decoder_3/Shape:output:0:model_2/transformer_decoder_3/strided_slice/stack:output:0<model_2/transformer_decoder_3/strided_slice/stack_1:output:0<model_2/transformer_decoder_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
3model_2/transformer_decoder_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
5model_2/transformer_decoder_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5model_2/transformer_decoder_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-model_2/transformer_decoder_3/strided_slice_1StridedSlice,model_2/transformer_decoder_3/Shape:output:0<model_2/transformer_decoder_3/strided_slice_1/stack:output:0>model_2/transformer_decoder_3/strided_slice_1/stack_1:output:0>model_2/transformer_decoder_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)model_2/transformer_decoder_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : k
)model_2/transformer_decoder_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
#model_2/transformer_decoder_3/rangeRange2model_2/transformer_decoder_3/range/start:output:06model_2/transformer_decoder_3/strided_slice_1:output:02model_2/transformer_decoder_3/range/delta:output:0*
_output_shapes
: ?
3model_2/transformer_decoder_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
5model_2/transformer_decoder_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ?
5model_2/transformer_decoder_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
-model_2/transformer_decoder_3/strided_slice_2StridedSlice,model_2/transformer_decoder_3/range:output:0<model_2/transformer_decoder_3/strided_slice_2/stack:output:0>model_2/transformer_decoder_3/strided_slice_2/stack_1:output:0>model_2/transformer_decoder_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maskm
+model_2/transformer_decoder_3/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : m
+model_2/transformer_decoder_3/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
%model_2/transformer_decoder_3/range_1Range4model_2/transformer_decoder_3/range_1/start:output:06model_2/transformer_decoder_3/strided_slice_1:output:04model_2/transformer_decoder_3/range_1/delta:output:0*
_output_shapes
: ?
*model_2/transformer_decoder_3/GreaterEqualGreaterEqual6model_2/transformer_decoder_3/strided_slice_2:output:0.model_2/transformer_decoder_3/range_1:output:0*
T0*
_output_shapes

:  ?
"model_2/transformer_decoder_3/CastCast.model_2/transformer_decoder_3/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  }
3model_2/transformer_decoder_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
5model_2/transformer_decoder_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5model_2/transformer_decoder_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-model_2/transformer_decoder_3/strided_slice_3StridedSlice,model_2/transformer_decoder_3/Shape:output:0<model_2/transformer_decoder_3/strided_slice_3/stack:output:0>model_2/transformer_decoder_3/strided_slice_3/stack_1:output:0>model_2/transformer_decoder_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
3model_2/transformer_decoder_3/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
5model_2/transformer_decoder_3/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5model_2/transformer_decoder_3/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-model_2/transformer_decoder_3/strided_slice_4StridedSlice,model_2/transformer_decoder_3/Shape:output:0<model_2/transformer_decoder_3/strided_slice_4/stack:output:0>model_2/transformer_decoder_3/strided_slice_4/stack_1:output:0>model_2/transformer_decoder_3/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-model_2/transformer_decoder_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
+model_2/transformer_decoder_3/Reshape/shapePack6model_2/transformer_decoder_3/Reshape/shape/0:output:06model_2/transformer_decoder_3/strided_slice_3:output:06model_2/transformer_decoder_3/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
%model_2/transformer_decoder_3/ReshapeReshape&model_2/transformer_decoder_3/Cast:y:04model_2/transformer_decoder_3/Reshape/shape:output:0*
T0*"
_output_shapes
:  w
,model_2/transformer_decoder_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(model_2/transformer_decoder_3/ExpandDims
ExpandDims4model_2/transformer_decoder_3/strided_slice:output:05model_2/transformer_decoder_3/ExpandDims/dim:output:0*
T0*
_output_shapes
:t
#model_2/transformer_decoder_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      k
)model_2/transformer_decoder_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$model_2/transformer_decoder_3/concatConcatV21model_2/transformer_decoder_3/ExpandDims:output:0,model_2/transformer_decoder_3/Const:output:02model_2/transformer_decoder_3/concat/axis:output:0*
N*
T0*
_output_shapes
:?
"model_2/transformer_decoder_3/TileTile.model_2/transformer_decoder_3/Reshape:output:0-model_2/transformer_decoder_3/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Umodel_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_decoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fmodel_2/transformer_decoder_3/multi_head_attention/query/einsum/EinsumEinsumGmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0]model_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Kmodel_2/transformer_decoder_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_decoder_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
<model_2/transformer_decoder_3/multi_head_attention/query/addAddV2Omodel_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum:output:0Smodel_2/transformer_decoder_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Smodel_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_decoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Dmodel_2/transformer_decoder_3/multi_head_attention/key/einsum/EinsumEinsumGmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0[model_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Imodel_2/transformer_decoder_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_decoder_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
:model_2/transformer_decoder_3/multi_head_attention/key/addAddV2Mmodel_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum:output:0Qmodel_2/transformer_decoder_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Umodel_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_decoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Fmodel_2/transformer_decoder_3/multi_head_attention/value/einsum/EinsumEinsumGmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0]model_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Kmodel_2/transformer_decoder_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_decoder_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
<model_2/transformer_decoder_3/multi_head_attention/value/addAddV2Omodel_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum:output:0Smodel_2/transformer_decoder_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? }
8model_2/transformer_decoder_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
6model_2/transformer_decoder_3/multi_head_attention/MulMul@model_2/transformer_decoder_3/multi_head_attention/query/add:z:0Amodel_2/transformer_decoder_3/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
@model_2/transformer_decoder_3/multi_head_attention/einsum/EinsumEinsum>model_2/transformer_decoder_3/multi_head_attention/key/add:z:0:model_2/transformer_decoder_3/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
Amodel_2/transformer_decoder_3/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
=model_2/transformer_decoder_3/multi_head_attention/ExpandDims
ExpandDims+model_2/transformer_decoder_3/Tile:output:0Jmodel_2/transformer_decoder_3/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
?model_2/transformer_decoder_3/multi_head_attention/softmax/CastCastFmodel_2/transformer_decoder_3/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  ?
@model_2/transformer_decoder_3/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
>model_2/transformer_decoder_3/multi_head_attention/softmax/subSubImodel_2/transformer_decoder_3/multi_head_attention/softmax/sub/x:output:0Cmodel_2/transformer_decoder_3/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
@model_2/transformer_decoder_3/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
>model_2/transformer_decoder_3/multi_head_attention/softmax/mulMulBmodel_2/transformer_decoder_3/multi_head_attention/softmax/sub:z:0Imodel_2/transformer_decoder_3/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
>model_2/transformer_decoder_3/multi_head_attention/softmax/addAddV2Imodel_2/transformer_decoder_3/multi_head_attention/einsum/Einsum:output:0Bmodel_2/transformer_decoder_3/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
Bmodel_2/transformer_decoder_3/multi_head_attention/softmax/SoftmaxSoftmaxBmodel_2/transformer_decoder_3/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
Emodel_2/transformer_decoder_3/multi_head_attention/dropout_2/IdentityIdentityLmodel_2/transformer_decoder_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
Bmodel_2/transformer_decoder_3/multi_head_attention/einsum_1/EinsumEinsumNmodel_2/transformer_decoder_3/multi_head_attention/dropout_2/Identity:output:0@model_2/transformer_decoder_3/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
`model_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_decoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Qmodel_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/EinsumEinsumKmodel_2/transformer_decoder_3/multi_head_attention/einsum_1/Einsum:output:0hmodel_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Vmodel_2/transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_decoder_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
Gmodel_2/transformer_decoder_3/multi_head_attention/attention_output/addAddV2Zmodel_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum:output:0^model_2/transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
.model_2/transformer_decoder_3/dropout/IdentityIdentityKmodel_2/transformer_decoder_3/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
!model_2/transformer_decoder_3/addAddV27model_2/transformer_decoder_3/dropout/Identity:output:0Gmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ?
Pmodel_2/transformer_decoder_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
>model_2/transformer_decoder_3/layer_normalization/moments/meanMean%model_2/transformer_decoder_3/add:z:0Ymodel_2/transformer_decoder_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Fmodel_2/transformer_decoder_3/layer_normalization/moments/StopGradientStopGradientGmodel_2/transformer_decoder_3/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Kmodel_2/transformer_decoder_3/layer_normalization/moments/SquaredDifferenceSquaredDifference%model_2/transformer_decoder_3/add:z:0Omodel_2/transformer_decoder_3/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Tmodel_2/transformer_decoder_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Bmodel_2/transformer_decoder_3/layer_normalization/moments/varianceMeanOmodel_2/transformer_decoder_3/layer_normalization/moments/SquaredDifference:z:0]model_2/transformer_decoder_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Amodel_2/transformer_decoder_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
?model_2/transformer_decoder_3/layer_normalization/batchnorm/addAddV2Kmodel_2/transformer_decoder_3/layer_normalization/moments/variance:output:0Jmodel_2/transformer_decoder_3/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_decoder_3/layer_normalization/batchnorm/RsqrtRsqrtCmodel_2/transformer_decoder_3/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Nmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_decoder_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_2/transformer_decoder_3/layer_normalization/batchnorm/mulMulEmodel_2/transformer_decoder_3/layer_normalization/batchnorm/Rsqrt:y:0Vmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul_1Mul%model_2/transformer_decoder_3/add:z:0Cmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul_2MulGmodel_2/transformer_decoder_3/layer_normalization/moments/mean:output:0Cmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Jmodel_2/transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_decoder_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
?model_2/transformer_decoder_3/layer_normalization/batchnorm/subSubRmodel_2/transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Amodel_2/transformer_decoder_3/layer_normalization/batchnorm/add_1AddV2Emodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul_1:z:0Cmodel_2/transformer_decoder_3/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
<model_2/transformer_decoder_3/dense/Tensordot/ReadVariableOpReadVariableOpEmodel_2_transformer_decoder_3_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0|
2model_2/transformer_decoder_3/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
2model_2/transformer_decoder_3/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
3model_2/transformer_decoder_3/dense/Tensordot/ShapeShapeEmodel_2/transformer_decoder_3/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:}
;model_2/transformer_decoder_3/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_2/transformer_decoder_3/dense/Tensordot/GatherV2GatherV2<model_2/transformer_decoder_3/dense/Tensordot/Shape:output:0;model_2/transformer_decoder_3/dense/Tensordot/free:output:0Dmodel_2/transformer_decoder_3/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
=model_2/transformer_decoder_3/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_2/transformer_decoder_3/dense/Tensordot/GatherV2_1GatherV2<model_2/transformer_decoder_3/dense/Tensordot/Shape:output:0;model_2/transformer_decoder_3/dense/Tensordot/axes:output:0Fmodel_2/transformer_decoder_3/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
3model_2/transformer_decoder_3/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
2model_2/transformer_decoder_3/dense/Tensordot/ProdProd?model_2/transformer_decoder_3/dense/Tensordot/GatherV2:output:0<model_2/transformer_decoder_3/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
5model_2/transformer_decoder_3/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
4model_2/transformer_decoder_3/dense/Tensordot/Prod_1ProdAmodel_2/transformer_decoder_3/dense/Tensordot/GatherV2_1:output:0>model_2/transformer_decoder_3/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: {
9model_2/transformer_decoder_3/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4model_2/transformer_decoder_3/dense/Tensordot/concatConcatV2;model_2/transformer_decoder_3/dense/Tensordot/free:output:0;model_2/transformer_decoder_3/dense/Tensordot/axes:output:0Bmodel_2/transformer_decoder_3/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
3model_2/transformer_decoder_3/dense/Tensordot/stackPack;model_2/transformer_decoder_3/dense/Tensordot/Prod:output:0=model_2/transformer_decoder_3/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
7model_2/transformer_decoder_3/dense/Tensordot/transpose	TransposeEmodel_2/transformer_decoder_3/layer_normalization/batchnorm/add_1:z:0=model_2/transformer_decoder_3/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
5model_2/transformer_decoder_3/dense/Tensordot/ReshapeReshape;model_2/transformer_decoder_3/dense/Tensordot/transpose:y:0<model_2/transformer_decoder_3/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
4model_2/transformer_decoder_3/dense/Tensordot/MatMulMatMul>model_2/transformer_decoder_3/dense/Tensordot/Reshape:output:0Dmodel_2/transformer_decoder_3/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
5model_2/transformer_decoder_3/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:}
;model_2/transformer_decoder_3/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_2/transformer_decoder_3/dense/Tensordot/concat_1ConcatV2?model_2/transformer_decoder_3/dense/Tensordot/GatherV2:output:0>model_2/transformer_decoder_3/dense/Tensordot/Const_2:output:0Dmodel_2/transformer_decoder_3/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
-model_2/transformer_decoder_3/dense/TensordotReshape>model_2/transformer_decoder_3/dense/Tensordot/MatMul:product:0?model_2/transformer_decoder_3/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
:model_2/transformer_decoder_3/dense/BiasAdd/ReadVariableOpReadVariableOpCmodel_2_transformer_decoder_3_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
+model_2/transformer_decoder_3/dense/BiasAddBiasAdd6model_2/transformer_decoder_3/dense/Tensordot:output:0Bmodel_2/transformer_decoder_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
(model_2/transformer_decoder_3/dense/ReluRelu4model_2/transformer_decoder_3/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
>model_2/transformer_decoder_3/dense_1/Tensordot/ReadVariableOpReadVariableOpGmodel_2_transformer_decoder_3_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0~
4model_2/transformer_decoder_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:?
4model_2/transformer_decoder_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
5model_2/transformer_decoder_3/dense_1/Tensordot/ShapeShape6model_2/transformer_decoder_3/dense/Relu:activations:0*
T0*
_output_shapes
:
=model_2/transformer_decoder_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_2/transformer_decoder_3/dense_1/Tensordot/GatherV2GatherV2>model_2/transformer_decoder_3/dense_1/Tensordot/Shape:output:0=model_2/transformer_decoder_3/dense_1/Tensordot/free:output:0Fmodel_2/transformer_decoder_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:?
?model_2/transformer_decoder_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
:model_2/transformer_decoder_3/dense_1/Tensordot/GatherV2_1GatherV2>model_2/transformer_decoder_3/dense_1/Tensordot/Shape:output:0=model_2/transformer_decoder_3/dense_1/Tensordot/axes:output:0Hmodel_2/transformer_decoder_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5model_2/transformer_decoder_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
4model_2/transformer_decoder_3/dense_1/Tensordot/ProdProdAmodel_2/transformer_decoder_3/dense_1/Tensordot/GatherV2:output:0>model_2/transformer_decoder_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: ?
7model_2/transformer_decoder_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
6model_2/transformer_decoder_3/dense_1/Tensordot/Prod_1ProdCmodel_2/transformer_decoder_3/dense_1/Tensordot/GatherV2_1:output:0@model_2/transformer_decoder_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: }
;model_2/transformer_decoder_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6model_2/transformer_decoder_3/dense_1/Tensordot/concatConcatV2=model_2/transformer_decoder_3/dense_1/Tensordot/free:output:0=model_2/transformer_decoder_3/dense_1/Tensordot/axes:output:0Dmodel_2/transformer_decoder_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5model_2/transformer_decoder_3/dense_1/Tensordot/stackPack=model_2/transformer_decoder_3/dense_1/Tensordot/Prod:output:0?model_2/transformer_decoder_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
9model_2/transformer_decoder_3/dense_1/Tensordot/transpose	Transpose6model_2/transformer_decoder_3/dense/Relu:activations:0?model_2/transformer_decoder_3/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
7model_2/transformer_decoder_3/dense_1/Tensordot/ReshapeReshape=model_2/transformer_decoder_3/dense_1/Tensordot/transpose:y:0>model_2/transformer_decoder_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
6model_2/transformer_decoder_3/dense_1/Tensordot/MatMulMatMul@model_2/transformer_decoder_3/dense_1/Tensordot/Reshape:output:0Fmodel_2/transformer_decoder_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
7model_2/transformer_decoder_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
=model_2/transformer_decoder_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
8model_2/transformer_decoder_3/dense_1/Tensordot/concat_1ConcatV2Amodel_2/transformer_decoder_3/dense_1/Tensordot/GatherV2:output:0@model_2/transformer_decoder_3/dense_1/Tensordot/Const_2:output:0Fmodel_2/transformer_decoder_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
/model_2/transformer_decoder_3/dense_1/TensordotReshape@model_2/transformer_decoder_3/dense_1/Tensordot/MatMul:product:0Amodel_2/transformer_decoder_3/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
<model_2/transformer_decoder_3/dense_1/BiasAdd/ReadVariableOpReadVariableOpEmodel_2_transformer_decoder_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-model_2/transformer_decoder_3/dense_1/BiasAddBiasAdd8model_2/transformer_decoder_3/dense_1/Tensordot:output:0Dmodel_2/transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
0model_2/transformer_decoder_3/dropout_1/IdentityIdentity6model_2/transformer_decoder_3/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
#model_2/transformer_decoder_3/add_1AddV2Emodel_2/transformer_decoder_3/layer_normalization/batchnorm/add_1:z:09model_2/transformer_decoder_3/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Rmodel_2/transformer_decoder_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
@model_2/transformer_decoder_3/layer_normalization_1/moments/meanMean'model_2/transformer_decoder_3/add_1:z:0[model_2/transformer_decoder_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Hmodel_2/transformer_decoder_3/layer_normalization_1/moments/StopGradientStopGradientImodel_2/transformer_decoder_3/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Mmodel_2/transformer_decoder_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifference'model_2/transformer_decoder_3/add_1:z:0Qmodel_2/transformer_decoder_3/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Vmodel_2/transformer_decoder_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
Dmodel_2/transformer_decoder_3/layer_normalization_1/moments/varianceMeanQmodel_2/transformer_decoder_3/layer_normalization_1/moments/SquaredDifference:z:0_model_2/transformer_decoder_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
Cmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
Amodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/addAddV2Mmodel_2/transformer_decoder_3/layer_normalization_1/moments/variance:output:0Lmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/RsqrtRsqrtEmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Pmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpYmodel_2_transformer_decoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mulMulGmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/Rsqrt:y:0Xmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul_1Mul'model_2/transformer_decoder_3/add_1:z:0Emodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul_2MulImodel_2/transformer_decoder_3/layer_normalization_1/moments/mean:output:0Emodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Lmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpUmodel_2_transformer_decoder_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
Amodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/subSubTmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0Gmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
Cmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/add_1AddV2Gmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul_1:z:0Emodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? {
9model_2/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
'model_2/global_average_pooling1d_2/MeanMeanGmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/add_1:z:0Bmodel_2/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
%model_2/dense_4/MatMul/ReadVariableOpReadVariableOp.model_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model_2/dense_4/MatMulMatMul0model_2/global_average_pooling1d_2/Mean:output:0-model_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
&model_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_2/dense_4/BiasAddBiasAdd model_2/dense_4/MatMul:product:0.model_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@p
model_2/dense_4/ReluRelu model_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
%model_2/dense_5/MatMul/ReadVariableOpReadVariableOp.model_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
model_2/dense_5/MatMulMatMul"model_2/dense_4/Relu:activations:0-model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
&model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_2/dense_5/BiasAddBiasAdd model_2/dense_5/MatMul:product:0.model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? v
model_2/dense_5/SoftmaxSoftmax model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? p
IdentityIdentity!model_2/dense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp'^model_2/dense_4/BiasAdd/ReadVariableOp&^model_2/dense_4/MatMul/ReadVariableOp'^model_2/dense_5/BiasAdd/ReadVariableOp&^model_2/dense_5/MatMul/ReadVariableOp%^model_2/embedding_3/embedding_lookupM^model_2/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2J^model_2/token_and_position_embedding_2/position_embedding3/ReadVariableOpI^model_2/token_and_position_embedding_2/token_embedding3/embedding_lookup;^model_2/transformer_decoder_3/dense/BiasAdd/ReadVariableOp=^model_2/transformer_decoder_3/dense/Tensordot/ReadVariableOp=^model_2/transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp?^model_2/transformer_decoder_3/dense_1/Tensordot/ReadVariableOpK^model_2/transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpO^model_2/transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpM^model_2/transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpQ^model_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpW^model_2/transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpa^model_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_decoder_3/multi_head_attention/key/add/ReadVariableOpT^model_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_decoder_3/multi_head_attention/query/add/ReadVariableOpV^model_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_decoder_3/multi_head_attention/value/add/ReadVariableOpV^model_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp;^model_2/transformer_encoder_3/dense/BiasAdd/ReadVariableOp=^model_2/transformer_encoder_3/dense/Tensordot/ReadVariableOp=^model_2/transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp?^model_2/transformer_encoder_3/dense_1/Tensordot/ReadVariableOpK^model_2/transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpO^model_2/transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpM^model_2/transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpQ^model_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpW^model_2/transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpa^model_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_encoder_3/multi_head_attention/key/add/ReadVariableOpT^model_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_encoder_3/multi_head_attention/query/add/ReadVariableOpV^model_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_encoder_3/multi_head_attention/value/add/ReadVariableOpV^model_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_2/dense_4/BiasAdd/ReadVariableOp&model_2/dense_4/BiasAdd/ReadVariableOp2N
%model_2/dense_4/MatMul/ReadVariableOp%model_2/dense_4/MatMul/ReadVariableOp2P
&model_2/dense_5/BiasAdd/ReadVariableOp&model_2/dense_5/BiasAdd/ReadVariableOp2N
%model_2/dense_5/MatMul/ReadVariableOp%model_2/dense_5/MatMul/ReadVariableOp2L
$model_2/embedding_3/embedding_lookup$model_2/embedding_3/embedding_lookup2?
Lmodel_2/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Lmodel_2/text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22?
Imodel_2/token_and_position_embedding_2/position_embedding3/ReadVariableOpImodel_2/token_and_position_embedding_2/position_embedding3/ReadVariableOp2?
Hmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookupHmodel_2/token_and_position_embedding_2/token_embedding3/embedding_lookup2x
:model_2/transformer_decoder_3/dense/BiasAdd/ReadVariableOp:model_2/transformer_decoder_3/dense/BiasAdd/ReadVariableOp2|
<model_2/transformer_decoder_3/dense/Tensordot/ReadVariableOp<model_2/transformer_decoder_3/dense/Tensordot/ReadVariableOp2|
<model_2/transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp<model_2/transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp2?
>model_2/transformer_decoder_3/dense_1/Tensordot/ReadVariableOp>model_2/transformer_decoder_3/dense_1/Tensordot/ReadVariableOp2?
Jmodel_2/transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpJmodel_2/transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp2?
Nmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpNmodel_2/transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp2?
Lmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpLmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp2?
Pmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpPmodel_2/transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Vmodel_2/transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpVmodel_2/transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp2?
`model_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Imodel_2/transformer_decoder_3/multi_head_attention/key/add/ReadVariableOpImodel_2/transformer_decoder_3/multi_head_attention/key/add/ReadVariableOp2?
Smodel_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Kmodel_2/transformer_decoder_3/multi_head_attention/query/add/ReadVariableOpKmodel_2/transformer_decoder_3/multi_head_attention/query/add/ReadVariableOp2?
Umodel_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Kmodel_2/transformer_decoder_3/multi_head_attention/value/add/ReadVariableOpKmodel_2/transformer_decoder_3/multi_head_attention/value/add/ReadVariableOp2?
Umodel_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:model_2/transformer_encoder_3/dense/BiasAdd/ReadVariableOp:model_2/transformer_encoder_3/dense/BiasAdd/ReadVariableOp2|
<model_2/transformer_encoder_3/dense/Tensordot/ReadVariableOp<model_2/transformer_encoder_3/dense/Tensordot/ReadVariableOp2|
<model_2/transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp<model_2/transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp2?
>model_2/transformer_encoder_3/dense_1/Tensordot/ReadVariableOp>model_2/transformer_encoder_3/dense_1/Tensordot/ReadVariableOp2?
Jmodel_2/transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpJmodel_2/transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp2?
Nmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpNmodel_2/transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp2?
Lmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpLmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp2?
Pmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpPmodel_2/transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Vmodel_2/transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpVmodel_2/transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp2?
`model_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Imodel_2/transformer_encoder_3/multi_head_attention/key/add/ReadVariableOpImodel_2/transformer_encoder_3/multi_head_attention/key/add/ReadVariableOp2?
Smodel_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Kmodel_2/transformer_encoder_3/multi_head_attention/query/add/ReadVariableOpKmodel_2/transformer_encoder_3/multi_head_attention/query/add/ReadVariableOp2?
Umodel_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Kmodel_2/transformer_encoder_3/multi_head_attention/value/add/ReadVariableOpKmodel_2/transformer_encoder_3/multi_head_attention/value/add/ReadVariableOp2?
Umodel_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_namePhrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
Token_role:
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
?3
C__inference_model_2_layer_call_and_return_conditional_losses_335732
inputs_0
inputs_1U
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	Z
Gtoken_and_position_embedding_2_token_embedding3_embedding_lookup_335351:	?\
Jtoken_and_position_embedding_2_position_embedding3_readvariableop_resource: 5
#embedding_3_embedding_lookup_335375:l
Vtransformer_encoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:^
Ltransformer_encoder_3_multi_head_attention_query_add_readvariableop_resource:j
Ttransformer_encoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:\
Jtransformer_encoder_3_multi_head_attention_key_add_readvariableop_resource:l
Vtransformer_encoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:^
Ltransformer_encoder_3_multi_head_attention_value_add_readvariableop_resource:w
atransformer_encoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:e
Wtransformer_encoder_3_multi_head_attention_attention_output_add_readvariableop_resource:]
Otransformer_encoder_3_layer_normalization_batchnorm_mul_readvariableop_resource:Y
Ktransformer_encoder_3_layer_normalization_batchnorm_readvariableop_resource:O
=transformer_encoder_3_dense_tensordot_readvariableop_resource:I
;transformer_encoder_3_dense_biasadd_readvariableop_resource:Q
?transformer_encoder_3_dense_1_tensordot_readvariableop_resource:K
=transformer_encoder_3_dense_1_biasadd_readvariableop_resource:_
Qtransformer_encoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:[
Mtransformer_encoder_3_layer_normalization_1_batchnorm_readvariableop_resource:l
Vtransformer_decoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:^
Ltransformer_decoder_3_multi_head_attention_query_add_readvariableop_resource:j
Ttransformer_decoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:\
Jtransformer_decoder_3_multi_head_attention_key_add_readvariableop_resource:l
Vtransformer_decoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:^
Ltransformer_decoder_3_multi_head_attention_value_add_readvariableop_resource:w
atransformer_decoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:e
Wtransformer_decoder_3_multi_head_attention_attention_output_add_readvariableop_resource:]
Otransformer_decoder_3_layer_normalization_batchnorm_mul_readvariableop_resource:Y
Ktransformer_decoder_3_layer_normalization_batchnorm_readvariableop_resource:O
=transformer_decoder_3_dense_tensordot_readvariableop_resource:I
;transformer_decoder_3_dense_biasadd_readvariableop_resource:Q
?transformer_decoder_3_dense_1_tensordot_readvariableop_resource:K
=transformer_decoder_3_dense_1_biasadd_readvariableop_resource:_
Qtransformer_decoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:[
Mtransformer_decoder_3_layer_normalization_1_batchnorm_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@ 5
'dense_5_biasadd_readvariableop_resource: 
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_3/embedding_lookup?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?Atoken_and_position_embedding_2/position_embedding3/ReadVariableOp?@token_and_position_embedding_2/token_embedding3/embedding_lookup?2transformer_decoder_3/dense/BiasAdd/ReadVariableOp?4transformer_decoder_3/dense/Tensordot/ReadVariableOp?4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp?6transformer_decoder_3/dense_1/Tensordot/ReadVariableOp?Btransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp?Ftransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp?Dtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp?Htransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp?Ntransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp?Xtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_decoder_3/multi_head_attention/key/add/ReadVariableOp?Ktransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Ctransformer_decoder_3/multi_head_attention/query/add/ReadVariableOp?Mtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Ctransformer_decoder_3/multi_head_attention/value/add/ReadVariableOp?Mtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp?2transformer_encoder_3/dense/BiasAdd/ReadVariableOp?4transformer_encoder_3/dense/Tensordot/ReadVariableOp?4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp?6transformer_encoder_3/dense_1/Tensordot/ReadVariableOp?Btransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp?Ftransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp?Dtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp?Htransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp?Ntransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp?Xtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_encoder_3/multi_head_attention/key/add/ReadVariableOp?Ktransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Ctransformer_encoder_3/multi_head_attention/query/add/ReadVariableOp?Mtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Ctransformer_encoder_3/multi_head_attention/value/add/ReadVariableOp?Mtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp}
text_vectorization/SqueezeSqueezeinputs_0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
@token_and_position_embedding_2/token_embedding3/embedding_lookupResourceGatherGtoken_and_position_embedding_2_token_embedding3_embedding_lookup_335351?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*Z
_classP
NLloc:@token_and_position_embedding_2/token_embedding3/embedding_lookup/335351*+
_output_shapes
:????????? *
dtype0?
Itoken_and_position_embedding_2/token_embedding3/embedding_lookup/IdentityIdentityItoken_and_position_embedding_2/token_embedding3/embedding_lookup:output:0*
T0*Z
_classP
NLloc:@token_and_position_embedding_2/token_embedding3/embedding_lookup/335351*+
_output_shapes
:????????? ?
Ktoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1IdentityRtoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
8token_and_position_embedding_2/position_embedding3/ShapeShapeTtoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Ftoken_and_position_embedding_2/position_embedding3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Htoken_and_position_embedding_2/position_embedding3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Htoken_and_position_embedding_2/position_embedding3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@token_and_position_embedding_2/position_embedding3/strided_sliceStridedSliceAtoken_and_position_embedding_2/position_embedding3/Shape:output:0Otoken_and_position_embedding_2/position_embedding3/strided_slice/stack:output:0Qtoken_and_position_embedding_2/position_embedding3/strided_slice/stack_1:output:0Qtoken_and_position_embedding_2/position_embedding3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Atoken_and_position_embedding_2/position_embedding3/ReadVariableOpReadVariableOpJtoken_and_position_embedding_2_position_embedding3_readvariableop_resource*
_output_shapes

: *
dtype0z
8token_and_position_embedding_2/position_embedding3/ConstConst*
_output_shapes
: *
dtype0*
value	B : |
:token_and_position_embedding_2/position_embedding3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Jtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Htoken_and_position_embedding_2/position_embedding3/strided_slice_1/stackPackAtoken_and_position_embedding_2/position_embedding3/Const:output:0Stoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ltoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1PackItoken_and_position_embedding_2/position_embedding3/strided_slice:output:0Utoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ltoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Jtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2PackCtoken_and_position_embedding_2/position_embedding3/Const_1:output:0Utoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Btoken_and_position_embedding_2/position_embedding3/strided_slice_1StridedSliceItoken_and_position_embedding_2/position_embedding3/ReadVariableOp:value:0Qtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack:output:0Stoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1:output:0Stoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
>token_and_position_embedding_2/position_embedding3/BroadcastToBroadcastToKtoken_and_position_embedding_2/position_embedding3/strided_slice_1:output:0Atoken_and_position_embedding_2/position_embedding3/Shape:output:0*
T0*+
_output_shapes
:????????? ?
"token_and_position_embedding_2/addAddV2Ttoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1:output:0Gtoken_and_position_embedding_2/position_embedding3/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? c
embedding_3/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
embedding_3/embedding_lookupResourceGather#embedding_3_embedding_lookup_335375embedding_3/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_3/embedding_lookup/335375*+
_output_shapes
:????????? *
dtype0?
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_3/embedding_lookup/335375*+
_output_shapes
:????????? ?
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
	add_2/addAddV2&token_and_position_embedding_2/add:z:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:????????? ?
Mtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_encoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_encoder_3/multi_head_attention/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_encoder_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpLtransformer_encoder_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_encoder_3/multi_head_attention/query/addAddV2Gtransformer_encoder_3/multi_head_attention/query/einsum/Einsum:output:0Ktransformer_encoder_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ktransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
<transformer_encoder_3/multi_head_attention/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Atransformer_encoder_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpJtransformer_encoder_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
2transformer_encoder_3/multi_head_attention/key/addAddV2Etransformer_encoder_3/multi_head_attention/key/einsum/Einsum:output:0Itransformer_encoder_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Mtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_encoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_encoder_3/multi_head_attention/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_encoder_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpLtransformer_encoder_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_encoder_3/multi_head_attention/value/addAddV2Gtransformer_encoder_3/multi_head_attention/value/einsum/Einsum:output:0Ktransformer_encoder_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? u
0transformer_encoder_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
.transformer_encoder_3/multi_head_attention/MulMul8transformer_encoder_3/multi_head_attention/query/add:z:09transformer_encoder_3/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
8transformer_encoder_3/multi_head_attention/einsum/EinsumEinsum6transformer_encoder_3/multi_head_attention/key/add:z:02transformer_encoder_3/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
:transformer_encoder_3/multi_head_attention/softmax/SoftmaxSoftmaxAtransformer_encoder_3/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_encoder_3/multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
@transformer_encoder_3/multi_head_attention/dropout_2/dropout/MulMulDtransformer_encoder_3/multi_head_attention/softmax/Softmax:softmax:0Ktransformer_encoder_3/multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_encoder_3/multi_head_attention/dropout_2/dropout/ShapeShapeDtransformer_encoder_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Ytransformer_encoder_3/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniformKtransformer_encoder_3/multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0?
Ktransformer_encoder_3/multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Itransformer_encoder_3/multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualbtransformer_encoder_3/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0Ttransformer_encoder_3/multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
Atransformer_encoder_3/multi_head_attention/dropout_2/dropout/CastCastMtransformer_encoder_3/multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
Btransformer_encoder_3/multi_head_attention/dropout_2/dropout/Mul_1MulDtransformer_encoder_3/multi_head_attention/dropout_2/dropout/Mul:z:0Etransformer_encoder_3/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
:transformer_encoder_3/multi_head_attention/einsum_1/EinsumEinsumFtransformer_encoder_3/multi_head_attention/dropout_2/dropout/Mul_1:z:08transformer_encoder_3/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Xtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_encoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Itransformer_encoder_3/multi_head_attention/attention_output/einsum/EinsumEinsumCtransformer_encoder_3/multi_head_attention/einsum_1/Einsum:output:0`transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Ntransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWtransformer_encoder_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
?transformer_encoder_3/multi_head_attention/attention_output/addAddV2Rtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum:output:0Vtransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? p
+transformer_encoder_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
)transformer_encoder_3/dropout/dropout/MulMulCtransformer_encoder_3/multi_head_attention/attention_output/add:z:04transformer_encoder_3/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
+transformer_encoder_3/dropout/dropout/ShapeShapeCtransformer_encoder_3/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
Btransformer_encoder_3/dropout/dropout/random_uniform/RandomUniformRandomUniform4transformer_encoder_3/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0y
4transformer_encoder_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
2transformer_encoder_3/dropout/dropout/GreaterEqualGreaterEqualKtransformer_encoder_3/dropout/dropout/random_uniform/RandomUniform:output:0=transformer_encoder_3/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
*transformer_encoder_3/dropout/dropout/CastCast6transformer_encoder_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
+transformer_encoder_3/dropout/dropout/Mul_1Mul-transformer_encoder_3/dropout/dropout/Mul:z:0.transformer_encoder_3/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_3/addAddV2add_2/add:z:0/transformer_encoder_3/dropout/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_encoder_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_encoder_3/layer_normalization/moments/meanMeantransformer_encoder_3/add:z:0Qtransformer_encoder_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
>transformer_encoder_3/layer_normalization/moments/StopGradientStopGradient?transformer_encoder_3/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_encoder_3/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_3/add:z:0Gtransformer_encoder_3/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ltransformer_encoder_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_encoder_3/layer_normalization/moments/varianceMeanGtransformer_encoder_3/layer_normalization/moments/SquaredDifference:z:0Utransformer_encoder_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(~
9transformer_encoder_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
7transformer_encoder_3/layer_normalization/batchnorm/addAddV2Ctransformer_encoder_3/layer_normalization/moments/variance:output:0Btransformer_encoder_3/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/RsqrtRsqrt;transformer_encoder_3/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_encoder_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_encoder_3/layer_normalization/batchnorm/mulMul=transformer_encoder_3/layer_normalization/batchnorm/Rsqrt:y:0Ntransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/mul_1Multransformer_encoder_3/add:z:0;transformer_encoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/mul_2Mul?transformer_encoder_3/layer_normalization/moments/mean:output:0;transformer_encoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Btransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKtransformer_encoder_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_encoder_3/layer_normalization/batchnorm/subSubJtransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp:value:0=transformer_encoder_3/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/add_1AddV2=transformer_encoder_3/layer_normalization/batchnorm/mul_1:z:0;transformer_encoder_3/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
4transformer_encoder_3/dense/Tensordot/ReadVariableOpReadVariableOp=transformer_encoder_3_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0t
*transformer_encoder_3/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:{
*transformer_encoder_3/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
+transformer_encoder_3/dense/Tensordot/ShapeShape=transformer_encoder_3/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
3transformer_encoder_3/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_encoder_3/dense/Tensordot/GatherV2GatherV24transformer_encoder_3/dense/Tensordot/Shape:output:03transformer_encoder_3/dense/Tensordot/free:output:0<transformer_encoder_3/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
5transformer_encoder_3/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_encoder_3/dense/Tensordot/GatherV2_1GatherV24transformer_encoder_3/dense/Tensordot/Shape:output:03transformer_encoder_3/dense/Tensordot/axes:output:0>transformer_encoder_3/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
+transformer_encoder_3/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*transformer_encoder_3/dense/Tensordot/ProdProd7transformer_encoder_3/dense/Tensordot/GatherV2:output:04transformer_encoder_3/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: w
-transformer_encoder_3/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_encoder_3/dense/Tensordot/Prod_1Prod9transformer_encoder_3/dense/Tensordot/GatherV2_1:output:06transformer_encoder_3/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: s
1transformer_encoder_3/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,transformer_encoder_3/dense/Tensordot/concatConcatV23transformer_encoder_3/dense/Tensordot/free:output:03transformer_encoder_3/dense/Tensordot/axes:output:0:transformer_encoder_3/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+transformer_encoder_3/dense/Tensordot/stackPack3transformer_encoder_3/dense/Tensordot/Prod:output:05transformer_encoder_3/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
/transformer_encoder_3/dense/Tensordot/transpose	Transpose=transformer_encoder_3/layer_normalization/batchnorm/add_1:z:05transformer_encoder_3/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_encoder_3/dense/Tensordot/ReshapeReshape3transformer_encoder_3/dense/Tensordot/transpose:y:04transformer_encoder_3/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
,transformer_encoder_3/dense/Tensordot/MatMulMatMul6transformer_encoder_3/dense/Tensordot/Reshape:output:0<transformer_encoder_3/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
-transformer_encoder_3/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:u
3transformer_encoder_3/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_encoder_3/dense/Tensordot/concat_1ConcatV27transformer_encoder_3/dense/Tensordot/GatherV2:output:06transformer_encoder_3/dense/Tensordot/Const_2:output:0<transformer_encoder_3/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
%transformer_encoder_3/dense/TensordotReshape6transformer_encoder_3/dense/Tensordot/MatMul:product:07transformer_encoder_3/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
2transformer_encoder_3/dense/BiasAdd/ReadVariableOpReadVariableOp;transformer_encoder_3_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#transformer_encoder_3/dense/BiasAddBiasAdd.transformer_encoder_3/dense/Tensordot:output:0:transformer_encoder_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
 transformer_encoder_3/dense/ReluRelu,transformer_encoder_3/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
6transformer_encoder_3/dense_1/Tensordot/ReadVariableOpReadVariableOp?transformer_encoder_3_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0v
,transformer_encoder_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,transformer_encoder_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
-transformer_encoder_3/dense_1/Tensordot/ShapeShape.transformer_encoder_3/dense/Relu:activations:0*
T0*
_output_shapes
:w
5transformer_encoder_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_encoder_3/dense_1/Tensordot/GatherV2GatherV26transformer_encoder_3/dense_1/Tensordot/Shape:output:05transformer_encoder_3/dense_1/Tensordot/free:output:0>transformer_encoder_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7transformer_encoder_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2transformer_encoder_3/dense_1/Tensordot/GatherV2_1GatherV26transformer_encoder_3/dense_1/Tensordot/Shape:output:05transformer_encoder_3/dense_1/Tensordot/axes:output:0@transformer_encoder_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-transformer_encoder_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_encoder_3/dense_1/Tensordot/ProdProd9transformer_encoder_3/dense_1/Tensordot/GatherV2:output:06transformer_encoder_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/transformer_encoder_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
.transformer_encoder_3/dense_1/Tensordot/Prod_1Prod;transformer_encoder_3/dense_1/Tensordot/GatherV2_1:output:08transformer_encoder_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3transformer_encoder_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_encoder_3/dense_1/Tensordot/concatConcatV25transformer_encoder_3/dense_1/Tensordot/free:output:05transformer_encoder_3/dense_1/Tensordot/axes:output:0<transformer_encoder_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
-transformer_encoder_3/dense_1/Tensordot/stackPack5transformer_encoder_3/dense_1/Tensordot/Prod:output:07transformer_encoder_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
1transformer_encoder_3/dense_1/Tensordot/transpose	Transpose.transformer_encoder_3/dense/Relu:activations:07transformer_encoder_3/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
/transformer_encoder_3/dense_1/Tensordot/ReshapeReshape5transformer_encoder_3/dense_1/Tensordot/transpose:y:06transformer_encoder_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
.transformer_encoder_3/dense_1/Tensordot/MatMulMatMul8transformer_encoder_3/dense_1/Tensordot/Reshape:output:0>transformer_encoder_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y
/transformer_encoder_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w
5transformer_encoder_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_encoder_3/dense_1/Tensordot/concat_1ConcatV29transformer_encoder_3/dense_1/Tensordot/GatherV2:output:08transformer_encoder_3/dense_1/Tensordot/Const_2:output:0>transformer_encoder_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
'transformer_encoder_3/dense_1/TensordotReshape8transformer_encoder_3/dense_1/Tensordot/MatMul:product:09transformer_encoder_3/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp=transformer_encoder_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%transformer_encoder_3/dense_1/BiasAddBiasAdd0transformer_encoder_3/dense_1/Tensordot:output:0<transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? r
-transformer_encoder_3/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+transformer_encoder_3/dropout_1/dropout/MulMul.transformer_encoder_3/dense_1/BiasAdd:output:06transformer_encoder_3/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_encoder_3/dropout_1/dropout/ShapeShape.transformer_encoder_3/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
Dtransformer_encoder_3/dropout_1/dropout/random_uniform/RandomUniformRandomUniform6transformer_encoder_3/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0{
6transformer_encoder_3/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4transformer_encoder_3/dropout_1/dropout/GreaterEqualGreaterEqualMtransformer_encoder_3/dropout_1/dropout/random_uniform/RandomUniform:output:0?transformer_encoder_3/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
,transformer_encoder_3/dropout_1/dropout/CastCast8transformer_encoder_3/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
-transformer_encoder_3/dropout_1/dropout/Mul_1Mul/transformer_encoder_3/dropout_1/dropout/Mul:z:00transformer_encoder_3/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_3/add_1AddV2=transformer_encoder_3/layer_normalization/batchnorm/add_1:z:01transformer_encoder_3/dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ?
Jtransformer_encoder_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_encoder_3/layer_normalization_1/moments/meanMeantransformer_encoder_3/add_1:z:0Stransformer_encoder_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
@transformer_encoder_3/layer_normalization_1/moments/StopGradientStopGradientAtransformer_encoder_3/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Etransformer_encoder_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_3/add_1:z:0Itransformer_encoder_3/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ntransformer_encoder_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
<transformer_encoder_3/layer_normalization_1/moments/varianceMeanItransformer_encoder_3/layer_normalization_1/moments/SquaredDifference:z:0Wtransformer_encoder_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
;transformer_encoder_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
9transformer_encoder_3/layer_normalization_1/batchnorm/addAddV2Etransformer_encoder_3/layer_normalization_1/moments/variance:output:0Dtransformer_encoder_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/RsqrtRsqrt=transformer_encoder_3/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_encoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_encoder_3/layer_normalization_1/batchnorm/mulMul?transformer_encoder_3/layer_normalization_1/batchnorm/Rsqrt:y:0Ptransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/mul_1Multransformer_encoder_3/add_1:z:0=transformer_encoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/mul_2MulAtransformer_encoder_3/layer_normalization_1/moments/mean:output:0=transformer_encoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMtransformer_encoder_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_encoder_3/layer_normalization_1/batchnorm/subSubLtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0?transformer_encoder_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/add_1AddV2?transformer_encoder_3/layer_normalization_1/batchnorm/mul_1:z:0=transformer_encoder_3/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_3/ShapeShape?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:s
)transformer_decoder_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+transformer_decoder_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#transformer_decoder_3/strided_sliceStridedSlice$transformer_decoder_3/Shape:output:02transformer_decoder_3/strided_slice/stack:output:04transformer_decoder_3/strided_slice/stack_1:output:04transformer_decoder_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%transformer_decoder_3/strided_slice_1StridedSlice$transformer_decoder_3/Shape:output:04transformer_decoder_3/strided_slice_1/stack:output:06transformer_decoder_3/strided_slice_1/stack_1:output:06transformer_decoder_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!transformer_decoder_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!transformer_decoder_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_3/rangeRange*transformer_decoder_3/range/start:output:0.transformer_decoder_3/strided_slice_1:output:0*transformer_decoder_3/range/delta:output:0*
_output_shapes
: |
+transformer_decoder_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ~
-transformer_decoder_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ~
-transformer_decoder_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
%transformer_decoder_3/strided_slice_2StridedSlice$transformer_decoder_3/range:output:04transformer_decoder_3/strided_slice_2/stack:output:06transformer_decoder_3/strided_slice_2/stack_1:output:06transformer_decoder_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maske
#transformer_decoder_3/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : e
#transformer_decoder_3/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_3/range_1Range,transformer_decoder_3/range_1/start:output:0.transformer_decoder_3/strided_slice_1:output:0,transformer_decoder_3/range_1/delta:output:0*
_output_shapes
: ?
"transformer_decoder_3/GreaterEqualGreaterEqual.transformer_decoder_3/strided_slice_2:output:0&transformer_decoder_3/range_1:output:0*
T0*
_output_shapes

:  ?
transformer_decoder_3/CastCast&transformer_decoder_3/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  u
+transformer_decoder_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%transformer_decoder_3/strided_slice_3StridedSlice$transformer_decoder_3/Shape:output:04transformer_decoder_3/strided_slice_3/stack:output:06transformer_decoder_3/strided_slice_3/stack_1:output:06transformer_decoder_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_3/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%transformer_decoder_3/strided_slice_4StridedSlice$transformer_decoder_3/Shape:output:04transformer_decoder_3/strided_slice_4/stack:output:06transformer_decoder_3/strided_slice_4/stack_1:output:06transformer_decoder_3/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%transformer_decoder_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
#transformer_decoder_3/Reshape/shapePack.transformer_decoder_3/Reshape/shape/0:output:0.transformer_decoder_3/strided_slice_3:output:0.transformer_decoder_3/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_3/ReshapeReshapetransformer_decoder_3/Cast:y:0,transformer_decoder_3/Reshape/shape:output:0*
T0*"
_output_shapes
:  o
$transformer_decoder_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 transformer_decoder_3/ExpandDims
ExpandDims,transformer_decoder_3/strided_slice:output:0-transformer_decoder_3/ExpandDims/dim:output:0*
T0*
_output_shapes
:l
transformer_decoder_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      c
!transformer_decoder_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
transformer_decoder_3/concatConcatV2)transformer_decoder_3/ExpandDims:output:0$transformer_decoder_3/Const:output:0*transformer_decoder_3/concat/axis:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_3/TileTile&transformer_decoder_3/Reshape:output:0%transformer_decoder_3/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Mtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_decoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_decoder_3/multi_head_attention/query/einsum/EinsumEinsum?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0Utransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_decoder_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpLtransformer_decoder_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_decoder_3/multi_head_attention/query/addAddV2Gtransformer_decoder_3/multi_head_attention/query/einsum/Einsum:output:0Ktransformer_decoder_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ktransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_decoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
<transformer_decoder_3/multi_head_attention/key/einsum/EinsumEinsum?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0Stransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Atransformer_decoder_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpJtransformer_decoder_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
2transformer_decoder_3/multi_head_attention/key/addAddV2Etransformer_decoder_3/multi_head_attention/key/einsum/Einsum:output:0Itransformer_decoder_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Mtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_decoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_decoder_3/multi_head_attention/value/einsum/EinsumEinsum?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0Utransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_decoder_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpLtransformer_decoder_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_decoder_3/multi_head_attention/value/addAddV2Gtransformer_decoder_3/multi_head_attention/value/einsum/Einsum:output:0Ktransformer_decoder_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? u
0transformer_decoder_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
.transformer_decoder_3/multi_head_attention/MulMul8transformer_decoder_3/multi_head_attention/query/add:z:09transformer_decoder_3/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
8transformer_decoder_3/multi_head_attention/einsum/EinsumEinsum6transformer_decoder_3/multi_head_attention/key/add:z:02transformer_decoder_3/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
9transformer_decoder_3/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5transformer_decoder_3/multi_head_attention/ExpandDims
ExpandDims#transformer_decoder_3/Tile:output:0Btransformer_decoder_3/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
7transformer_decoder_3/multi_head_attention/softmax/CastCast>transformer_decoder_3/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  }
8transformer_decoder_3/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
6transformer_decoder_3/multi_head_attention/softmax/subSubAtransformer_decoder_3/multi_head_attention/softmax/sub/x:output:0;transformer_decoder_3/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  }
8transformer_decoder_3/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
6transformer_decoder_3/multi_head_attention/softmax/mulMul:transformer_decoder_3/multi_head_attention/softmax/sub:z:0Atransformer_decoder_3/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
6transformer_decoder_3/multi_head_attention/softmax/addAddV2Atransformer_decoder_3/multi_head_attention/einsum/Einsum:output:0:transformer_decoder_3/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
:transformer_decoder_3/multi_head_attention/softmax/SoftmaxSoftmax:transformer_decoder_3/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_decoder_3/multi_head_attention/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
@transformer_decoder_3/multi_head_attention/dropout_2/dropout/MulMulDtransformer_decoder_3/multi_head_attention/softmax/Softmax:softmax:0Ktransformer_decoder_3/multi_head_attention/dropout_2/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  ?
Btransformer_decoder_3/multi_head_attention/dropout_2/dropout/ShapeShapeDtransformer_decoder_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:?
Ytransformer_decoder_3/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniformRandomUniformKtransformer_decoder_3/multi_head_attention/dropout_2/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  *
dtype0?
Ktransformer_decoder_3/multi_head_attention/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
Itransformer_decoder_3/multi_head_attention/dropout_2/dropout/GreaterEqualGreaterEqualbtransformer_decoder_3/multi_head_attention/dropout_2/dropout/random_uniform/RandomUniform:output:0Ttransformer_decoder_3/multi_head_attention/dropout_2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  ?
Atransformer_decoder_3/multi_head_attention/dropout_2/dropout/CastCastMtransformer_decoder_3/multi_head_attention/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  ?
Btransformer_decoder_3/multi_head_attention/dropout_2/dropout/Mul_1MulDtransformer_decoder_3/multi_head_attention/dropout_2/dropout/Mul:z:0Etransformer_decoder_3/multi_head_attention/dropout_2/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  ?
:transformer_decoder_3/multi_head_attention/einsum_1/EinsumEinsumFtransformer_decoder_3/multi_head_attention/dropout_2/dropout/Mul_1:z:08transformer_decoder_3/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Xtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_decoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Itransformer_decoder_3/multi_head_attention/attention_output/einsum/EinsumEinsumCtransformer_decoder_3/multi_head_attention/einsum_1/Einsum:output:0`transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Ntransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWtransformer_decoder_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
?transformer_decoder_3/multi_head_attention/attention_output/addAddV2Rtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum:output:0Vtransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? p
+transformer_decoder_3/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
)transformer_decoder_3/dropout/dropout/MulMulCtransformer_decoder_3/multi_head_attention/attention_output/add:z:04transformer_decoder_3/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
+transformer_decoder_3/dropout/dropout/ShapeShapeCtransformer_decoder_3/multi_head_attention/attention_output/add:z:0*
T0*
_output_shapes
:?
Btransformer_decoder_3/dropout/dropout/random_uniform/RandomUniformRandomUniform4transformer_decoder_3/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0y
4transformer_decoder_3/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
2transformer_decoder_3/dropout/dropout/GreaterEqualGreaterEqualKtransformer_decoder_3/dropout/dropout/random_uniform/RandomUniform:output:0=transformer_decoder_3/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
*transformer_decoder_3/dropout/dropout/CastCast6transformer_decoder_3/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
+transformer_decoder_3/dropout/dropout/Mul_1Mul-transformer_decoder_3/dropout/dropout/Mul:z:0.transformer_decoder_3/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_3/addAddV2/transformer_decoder_3/dropout/dropout/Mul_1:z:0?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_decoder_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_decoder_3/layer_normalization/moments/meanMeantransformer_decoder_3/add:z:0Qtransformer_decoder_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
>transformer_decoder_3/layer_normalization/moments/StopGradientStopGradient?transformer_decoder_3/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_decoder_3/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_3/add:z:0Gtransformer_decoder_3/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ltransformer_decoder_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_decoder_3/layer_normalization/moments/varianceMeanGtransformer_decoder_3/layer_normalization/moments/SquaredDifference:z:0Utransformer_decoder_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(~
9transformer_decoder_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
7transformer_decoder_3/layer_normalization/batchnorm/addAddV2Ctransformer_decoder_3/layer_normalization/moments/variance:output:0Btransformer_decoder_3/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/RsqrtRsqrt;transformer_decoder_3/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_decoder_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_decoder_3/layer_normalization/batchnorm/mulMul=transformer_decoder_3/layer_normalization/batchnorm/Rsqrt:y:0Ntransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/mul_1Multransformer_decoder_3/add:z:0;transformer_decoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/mul_2Mul?transformer_decoder_3/layer_normalization/moments/mean:output:0;transformer_decoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Btransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKtransformer_decoder_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_decoder_3/layer_normalization/batchnorm/subSubJtransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp:value:0=transformer_decoder_3/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/add_1AddV2=transformer_decoder_3/layer_normalization/batchnorm/mul_1:z:0;transformer_decoder_3/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
4transformer_decoder_3/dense/Tensordot/ReadVariableOpReadVariableOp=transformer_decoder_3_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0t
*transformer_decoder_3/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:{
*transformer_decoder_3/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
+transformer_decoder_3/dense/Tensordot/ShapeShape=transformer_decoder_3/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
3transformer_decoder_3/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_decoder_3/dense/Tensordot/GatherV2GatherV24transformer_decoder_3/dense/Tensordot/Shape:output:03transformer_decoder_3/dense/Tensordot/free:output:0<transformer_decoder_3/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
5transformer_decoder_3/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_decoder_3/dense/Tensordot/GatherV2_1GatherV24transformer_decoder_3/dense/Tensordot/Shape:output:03transformer_decoder_3/dense/Tensordot/axes:output:0>transformer_decoder_3/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
+transformer_decoder_3/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*transformer_decoder_3/dense/Tensordot/ProdProd7transformer_decoder_3/dense/Tensordot/GatherV2:output:04transformer_decoder_3/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: w
-transformer_decoder_3/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_decoder_3/dense/Tensordot/Prod_1Prod9transformer_decoder_3/dense/Tensordot/GatherV2_1:output:06transformer_decoder_3/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: s
1transformer_decoder_3/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,transformer_decoder_3/dense/Tensordot/concatConcatV23transformer_decoder_3/dense/Tensordot/free:output:03transformer_decoder_3/dense/Tensordot/axes:output:0:transformer_decoder_3/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+transformer_decoder_3/dense/Tensordot/stackPack3transformer_decoder_3/dense/Tensordot/Prod:output:05transformer_decoder_3/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
/transformer_decoder_3/dense/Tensordot/transpose	Transpose=transformer_decoder_3/layer_normalization/batchnorm/add_1:z:05transformer_decoder_3/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_decoder_3/dense/Tensordot/ReshapeReshape3transformer_decoder_3/dense/Tensordot/transpose:y:04transformer_decoder_3/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
,transformer_decoder_3/dense/Tensordot/MatMulMatMul6transformer_decoder_3/dense/Tensordot/Reshape:output:0<transformer_decoder_3/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
-transformer_decoder_3/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:u
3transformer_decoder_3/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_decoder_3/dense/Tensordot/concat_1ConcatV27transformer_decoder_3/dense/Tensordot/GatherV2:output:06transformer_decoder_3/dense/Tensordot/Const_2:output:0<transformer_decoder_3/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
%transformer_decoder_3/dense/TensordotReshape6transformer_decoder_3/dense/Tensordot/MatMul:product:07transformer_decoder_3/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
2transformer_decoder_3/dense/BiasAdd/ReadVariableOpReadVariableOp;transformer_decoder_3_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#transformer_decoder_3/dense/BiasAddBiasAdd.transformer_decoder_3/dense/Tensordot:output:0:transformer_decoder_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
 transformer_decoder_3/dense/ReluRelu,transformer_decoder_3/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
6transformer_decoder_3/dense_1/Tensordot/ReadVariableOpReadVariableOp?transformer_decoder_3_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0v
,transformer_decoder_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,transformer_decoder_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
-transformer_decoder_3/dense_1/Tensordot/ShapeShape.transformer_decoder_3/dense/Relu:activations:0*
T0*
_output_shapes
:w
5transformer_decoder_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_decoder_3/dense_1/Tensordot/GatherV2GatherV26transformer_decoder_3/dense_1/Tensordot/Shape:output:05transformer_decoder_3/dense_1/Tensordot/free:output:0>transformer_decoder_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7transformer_decoder_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2transformer_decoder_3/dense_1/Tensordot/GatherV2_1GatherV26transformer_decoder_3/dense_1/Tensordot/Shape:output:05transformer_decoder_3/dense_1/Tensordot/axes:output:0@transformer_decoder_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-transformer_decoder_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_decoder_3/dense_1/Tensordot/ProdProd9transformer_decoder_3/dense_1/Tensordot/GatherV2:output:06transformer_decoder_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/transformer_decoder_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
.transformer_decoder_3/dense_1/Tensordot/Prod_1Prod;transformer_decoder_3/dense_1/Tensordot/GatherV2_1:output:08transformer_decoder_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3transformer_decoder_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_decoder_3/dense_1/Tensordot/concatConcatV25transformer_decoder_3/dense_1/Tensordot/free:output:05transformer_decoder_3/dense_1/Tensordot/axes:output:0<transformer_decoder_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
-transformer_decoder_3/dense_1/Tensordot/stackPack5transformer_decoder_3/dense_1/Tensordot/Prod:output:07transformer_decoder_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
1transformer_decoder_3/dense_1/Tensordot/transpose	Transpose.transformer_decoder_3/dense/Relu:activations:07transformer_decoder_3/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
/transformer_decoder_3/dense_1/Tensordot/ReshapeReshape5transformer_decoder_3/dense_1/Tensordot/transpose:y:06transformer_decoder_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
.transformer_decoder_3/dense_1/Tensordot/MatMulMatMul8transformer_decoder_3/dense_1/Tensordot/Reshape:output:0>transformer_decoder_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y
/transformer_decoder_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w
5transformer_decoder_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_decoder_3/dense_1/Tensordot/concat_1ConcatV29transformer_decoder_3/dense_1/Tensordot/GatherV2:output:08transformer_decoder_3/dense_1/Tensordot/Const_2:output:0>transformer_decoder_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
'transformer_decoder_3/dense_1/TensordotReshape8transformer_decoder_3/dense_1/Tensordot/MatMul:product:09transformer_decoder_3/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp=transformer_decoder_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%transformer_decoder_3/dense_1/BiasAddBiasAdd0transformer_decoder_3/dense_1/Tensordot:output:0<transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? r
-transformer_decoder_3/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
+transformer_decoder_3/dropout_1/dropout/MulMul.transformer_decoder_3/dense_1/BiasAdd:output:06transformer_decoder_3/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_decoder_3/dropout_1/dropout/ShapeShape.transformer_decoder_3/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:?
Dtransformer_decoder_3/dropout_1/dropout/random_uniform/RandomUniformRandomUniform6transformer_decoder_3/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0{
6transformer_decoder_3/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
4transformer_decoder_3/dropout_1/dropout/GreaterEqualGreaterEqualMtransformer_decoder_3/dropout_1/dropout/random_uniform/RandomUniform:output:0?transformer_decoder_3/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? ?
,transformer_decoder_3/dropout_1/dropout/CastCast8transformer_decoder_3/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? ?
-transformer_decoder_3/dropout_1/dropout/Mul_1Mul/transformer_decoder_3/dropout_1/dropout/Mul:z:00transformer_decoder_3/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_3/add_1AddV2=transformer_decoder_3/layer_normalization/batchnorm/add_1:z:01transformer_decoder_3/dropout_1/dropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? ?
Jtransformer_decoder_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_decoder_3/layer_normalization_1/moments/meanMeantransformer_decoder_3/add_1:z:0Stransformer_decoder_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
@transformer_decoder_3/layer_normalization_1/moments/StopGradientStopGradientAtransformer_decoder_3/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Etransformer_decoder_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_3/add_1:z:0Itransformer_decoder_3/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ntransformer_decoder_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
<transformer_decoder_3/layer_normalization_1/moments/varianceMeanItransformer_decoder_3/layer_normalization_1/moments/SquaredDifference:z:0Wtransformer_decoder_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
;transformer_decoder_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
9transformer_decoder_3/layer_normalization_1/batchnorm/addAddV2Etransformer_decoder_3/layer_normalization_1/moments/variance:output:0Dtransformer_decoder_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/RsqrtRsqrt=transformer_decoder_3/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_decoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_decoder_3/layer_normalization_1/batchnorm/mulMul?transformer_decoder_3/layer_normalization_1/batchnorm/Rsqrt:y:0Ptransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/mul_1Multransformer_decoder_3/add_1:z:0=transformer_decoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/mul_2MulAtransformer_decoder_3/layer_normalization_1/moments/mean:output:0=transformer_decoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMtransformer_decoder_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_decoder_3/layer_normalization_1/batchnorm/subSubLtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0?transformer_decoder_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/add_1AddV2?transformer_decoder_3/layer_normalization_1/batchnorm/mul_1:z:0=transformer_decoder_3/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? s
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_2/MeanMean?transformer_decoder_3/layer_normalization_1/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_4/MatMulMatMul(global_average_pooling1d_2/Mean:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? h
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_3/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2B^token_and_position_embedding_2/position_embedding3/ReadVariableOpA^token_and_position_embedding_2/token_embedding3/embedding_lookup3^transformer_decoder_3/dense/BiasAdd/ReadVariableOp5^transformer_decoder_3/dense/Tensordot/ReadVariableOp5^transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp7^transformer_decoder_3/dense_1/Tensordot/ReadVariableOpC^transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpG^transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpE^transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpI^transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpO^transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpY^transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^transformer_decoder_3/multi_head_attention/key/add/ReadVariableOpL^transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^transformer_decoder_3/multi_head_attention/query/add/ReadVariableOpN^transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^transformer_decoder_3/multi_head_attention/value/add/ReadVariableOpN^transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp3^transformer_encoder_3/dense/BiasAdd/ReadVariableOp5^transformer_encoder_3/dense/Tensordot/ReadVariableOp5^transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp7^transformer_encoder_3/dense_1/Tensordot/ReadVariableOpC^transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpG^transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpE^transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpI^transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpO^transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpY^transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^transformer_encoder_3/multi_head_attention/key/add/ReadVariableOpL^transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^transformer_encoder_3/multi_head_attention/query/add/ReadVariableOpN^transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^transformer_encoder_3/multi_head_attention/value/add/ReadVariableOpN^transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22?
Atoken_and_position_embedding_2/position_embedding3/ReadVariableOpAtoken_and_position_embedding_2/position_embedding3/ReadVariableOp2?
@token_and_position_embedding_2/token_embedding3/embedding_lookup@token_and_position_embedding_2/token_embedding3/embedding_lookup2h
2transformer_decoder_3/dense/BiasAdd/ReadVariableOp2transformer_decoder_3/dense/BiasAdd/ReadVariableOp2l
4transformer_decoder_3/dense/Tensordot/ReadVariableOp4transformer_decoder_3/dense/Tensordot/ReadVariableOp2l
4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp2p
6transformer_decoder_3/dense_1/Tensordot/ReadVariableOp6transformer_decoder_3/dense_1/Tensordot/ReadVariableOp2?
Btransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpBtransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp2?
Ftransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpFtransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp2?
Dtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpDtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp2?
Htransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpHtransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Ntransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpNtransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp2?
Xtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_decoder_3/multi_head_attention/key/add/ReadVariableOpAtransformer_decoder_3/multi_head_attention/key/add/ReadVariableOp2?
Ktransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpKtransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_decoder_3/multi_head_attention/query/add/ReadVariableOpCtransformer_decoder_3/multi_head_attention/query/add/ReadVariableOp2?
Mtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpMtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_decoder_3/multi_head_attention/value/add/ReadVariableOpCtransformer_decoder_3/multi_head_attention/value/add/ReadVariableOp2?
Mtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpMtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp2h
2transformer_encoder_3/dense/BiasAdd/ReadVariableOp2transformer_encoder_3/dense/BiasAdd/ReadVariableOp2l
4transformer_encoder_3/dense/Tensordot/ReadVariableOp4transformer_encoder_3/dense/Tensordot/ReadVariableOp2l
4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp2p
6transformer_encoder_3/dense_1/Tensordot/ReadVariableOp6transformer_encoder_3/dense_1/Tensordot/ReadVariableOp2?
Btransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpBtransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp2?
Ftransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpFtransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp2?
Dtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpDtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp2?
Htransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpHtransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Ntransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpNtransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp2?
Xtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_encoder_3/multi_head_attention/key/add/ReadVariableOpAtransformer_encoder_3/multi_head_attention/key/add/ReadVariableOp2?
Ktransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpKtransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_encoder_3/multi_head_attention/query/add/ReadVariableOpCtransformer_encoder_3/multi_head_attention/query/add/ReadVariableOp2?
Mtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpMtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_encoder_3/multi_head_attention/value/add/ReadVariableOpCtransformer_encoder_3/multi_head_attention/value/add/ReadVariableOp2?
Mtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpMtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:Q M
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
?
-
__inference__destroyer_336671
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
?
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_332798

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
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336393
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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
?
?
__inference__initializer_3366519
5key_value_init318608_lookuptableimportv2_table_handle1
-key_value_init318608_lookuptableimportv2_keys3
/key_value_init318608_lookuptableimportv2_values	
identity??(key_value_init318608/LookupTableImportV2?
(key_value_init318608/LookupTableImportV2LookupTableImportV25key_value_init318608_lookuptableimportv2_table_handle-key_value_init318608_lookuptableimportv2_keys/key_value_init318608_lookuptableimportv2_values*	
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
NoOpNoOp)^key_value_init318608/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2T
(key_value_init318608/LookupTableImportV2(key_value_init318608/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?

?
C__inference_dense_5_layer_call_and_return_conditional_losses_333308

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
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
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?

(__inference_model_2_layer_call_fn_334912
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24:

unknown_25: 

unknown_26:

unknown_27: 

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:@

unknown_39:@

unknown_40:@ 

unknown_41: 
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
unknown_41*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *I
_read_only_resource_inputs+
)'	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_334173o
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
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
??
?
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336587
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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
?
?
__inference_restore_fn_336698
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
value	B :L
IdentityIdentityConst:output:0^NoOp*
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
?A
?
__inference_adapt_step_202704
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
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
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
?
-
__inference__destroyer_336656
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
?
G
__inference__creator_336661
identity: ??MutableHashTable|
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_7*
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
??
?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_336146

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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
??
?
C__inference_model_2_layer_call_and_return_conditional_losses_334628

phrase

token_roleU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	8
%token_and_position_embedding_2_334541:	?7
%token_and_position_embedding_2_334543: $
embedding_3_334546:2
transformer_encoder_3_334550:.
transformer_encoder_3_334552:2
transformer_encoder_3_334554:.
transformer_encoder_3_334556:2
transformer_encoder_3_334558:.
transformer_encoder_3_334560:2
transformer_encoder_3_334562:*
transformer_encoder_3_334564:*
transformer_encoder_3_334566:*
transformer_encoder_3_334568:.
transformer_encoder_3_334570:*
transformer_encoder_3_334572:.
transformer_encoder_3_334574:*
transformer_encoder_3_334576:*
transformer_encoder_3_334578:*
transformer_encoder_3_334580:2
transformer_decoder_3_334583:.
transformer_decoder_3_334585:2
transformer_decoder_3_334587:.
transformer_decoder_3_334589:2
transformer_decoder_3_334591:.
transformer_decoder_3_334593:2
transformer_decoder_3_334595:*
transformer_decoder_3_334597:*
transformer_decoder_3_334599:*
transformer_decoder_3_334601:.
transformer_decoder_3_334603:*
transformer_decoder_3_334605:.
transformer_decoder_3_334607:*
transformer_decoder_3_334609:*
transformer_decoder_3_334611:*
transformer_decoder_3_334613: 
dense_4_334617:@
dense_4_334619:@ 
dense_5_334622:@ 
dense_5_334624: 
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_3/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?6token_and_position_embedding_2/StatefulPartitionedCall?-transformer_decoder_3/StatefulPartitionedCall?-transformer_encoder_3/StatefulPartitionedCall{
text_vectorization/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0%token_and_position_embedding_2_334541%token_and_position_embedding_2_334543*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_332883?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCall
token_roleembedding_3_334546*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_332899?
add_2/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_332909?
-transformer_encoder_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_encoder_3_334550transformer_encoder_3_334552transformer_encoder_3_334554transformer_encoder_3_334556transformer_encoder_3_334558transformer_encoder_3_334560transformer_encoder_3_334562transformer_encoder_3_334564transformer_encoder_3_334566transformer_encoder_3_334568transformer_encoder_3_334570transformer_encoder_3_334572transformer_encoder_3_334574transformer_encoder_3_334576transformer_encoder_3_334578transformer_encoder_3_334580*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333881?
-transformer_decoder_3/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_3/StatefulPartitionedCall:output:0transformer_decoder_3_334583transformer_decoder_3_334585transformer_decoder_3_334587transformer_decoder_3_334589transformer_decoder_3_334591transformer_decoder_3_334593transformer_decoder_3_334595transformer_decoder_3_334597transformer_decoder_3_334599transformer_decoder_3_334601transformer_decoder_3_334603transformer_decoder_3_334605transformer_decoder_3_334607transformer_decoder_3_334609transformer_decoder_3_334611transformer_decoder_3_334613*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333658?
*global_average_pooling1d_2/PartitionedCallPartitionedCall6transformer_decoder_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_332798?
dense_4/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_4_334617dense_4_334619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_333291?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_334622dense_5_334624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_333308w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV27^token_and_position_embedding_2/StatefulPartitionedCall.^transformer_decoder_3/StatefulPartitionedCall.^transformer_encoder_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2^
-transformer_decoder_3/StatefulPartitionedCall-transformer_decoder_3/StatefulPartitionedCall2^
-transformer_encoder_3/StatefulPartitionedCall-transformer_encoder_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namePhrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
Token_role:
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
6__inference_transformer_encoder_3_layer_call_fn_335834

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
	unknown_9:

unknown_10:

unknown_11:

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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333038s
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
?
?

(__inference_model_2_layer_call_fn_334820
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24:

unknown_25: 

unknown_26:

unknown_27: 

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:@

unknown_39:@

unknown_40:@ 

unknown_41: 
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
unknown_41*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *I
_read_only_resource_inputs+
)'	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_333315o
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
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
?

?
C__inference_dense_4_layer_call_and_return_conditional_losses_336618

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
?
R
&__inference_add_2_layer_call_fn_335791
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
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_332909d
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
?
W
;__inference_global_average_pooling1d_2_layer_call_fn_336592

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
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_332798i
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
?
?
(__inference_dense_4_layer_call_fn_336607

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
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_333291o
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
??
?k
"__inference__traced_restore_337518
file_prefix9
'assignvariableop_embedding_3_embeddings:3
!assignvariableop_1_dense_4_kernel:@-
assignvariableop_2_dense_4_bias:@3
!assignvariableop_3_dense_5_kernel:@ -
assignvariableop_4_dense_5_bias: `
Massignvariableop_5_token_and_position_embedding_2_token_embedding3_embeddings:	?b
Passignvariableop_6_token_and_position_embedding_2_position_embedding3_embeddings: `
Jassignvariableop_7_transformer_encoder_3_multi_head_attention_query_kernel:Z
Hassignvariableop_8_transformer_encoder_3_multi_head_attention_query_bias:^
Hassignvariableop_9_transformer_encoder_3_multi_head_attention_key_kernel:Y
Gassignvariableop_10_transformer_encoder_3_multi_head_attention_key_bias:a
Kassignvariableop_11_transformer_encoder_3_multi_head_attention_value_kernel:[
Iassignvariableop_12_transformer_encoder_3_multi_head_attention_value_bias:l
Vassignvariableop_13_transformer_encoder_3_multi_head_attention_attention_output_kernel:b
Tassignvariableop_14_transformer_encoder_3_multi_head_attention_attention_output_bias:Q
Cassignvariableop_15_transformer_encoder_3_layer_normalization_gamma:P
Bassignvariableop_16_transformer_encoder_3_layer_normalization_beta:S
Eassignvariableop_17_transformer_encoder_3_layer_normalization_1_gamma:R
Dassignvariableop_18_transformer_encoder_3_layer_normalization_1_beta:H
6assignvariableop_19_transformer_encoder_3_dense_kernel:B
4assignvariableop_20_transformer_encoder_3_dense_bias:J
8assignvariableop_21_transformer_encoder_3_dense_1_kernel:D
6assignvariableop_22_transformer_encoder_3_dense_1_bias:a
Kassignvariableop_23_transformer_decoder_3_multi_head_attention_query_kernel:[
Iassignvariableop_24_transformer_decoder_3_multi_head_attention_query_bias:_
Iassignvariableop_25_transformer_decoder_3_multi_head_attention_key_kernel:Y
Gassignvariableop_26_transformer_decoder_3_multi_head_attention_key_bias:a
Kassignvariableop_27_transformer_decoder_3_multi_head_attention_value_kernel:[
Iassignvariableop_28_transformer_decoder_3_multi_head_attention_value_bias:l
Vassignvariableop_29_transformer_decoder_3_multi_head_attention_attention_output_kernel:b
Tassignvariableop_30_transformer_decoder_3_multi_head_attention_attention_output_bias:Q
Cassignvariableop_31_transformer_decoder_3_layer_normalization_gamma:P
Bassignvariableop_32_transformer_decoder_3_layer_normalization_beta:S
Eassignvariableop_33_transformer_decoder_3_layer_normalization_1_gamma:R
Dassignvariableop_34_transformer_decoder_3_layer_normalization_1_beta:H
6assignvariableop_35_transformer_decoder_3_dense_kernel:B
4assignvariableop_36_transformer_decoder_3_dense_bias:J
8assignvariableop_37_transformer_decoder_3_dense_1_kernel:D
6assignvariableop_38_transformer_decoder_3_dense_1_bias:'
assignvariableop_39_adam_iter:	 )
assignvariableop_40_adam_beta_1: )
assignvariableop_41_adam_beta_2: (
assignvariableop_42_adam_decay: 0
&assignvariableop_43_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: %
assignvariableop_44_total_1: %
assignvariableop_45_count_1: #
assignvariableop_46_total: #
assignvariableop_47_count: C
1assignvariableop_48_adam_embedding_3_embeddings_m:;
)assignvariableop_49_adam_dense_4_kernel_m:@5
'assignvariableop_50_adam_dense_4_bias_m:@;
)assignvariableop_51_adam_dense_5_kernel_m:@ 5
'assignvariableop_52_adam_dense_5_bias_m: h
Uassignvariableop_53_adam_token_and_position_embedding_2_token_embedding3_embeddings_m:	?j
Xassignvariableop_54_adam_token_and_position_embedding_2_position_embedding3_embeddings_m: h
Rassignvariableop_55_adam_transformer_encoder_3_multi_head_attention_query_kernel_m:b
Passignvariableop_56_adam_transformer_encoder_3_multi_head_attention_query_bias_m:f
Passignvariableop_57_adam_transformer_encoder_3_multi_head_attention_key_kernel_m:`
Nassignvariableop_58_adam_transformer_encoder_3_multi_head_attention_key_bias_m:h
Rassignvariableop_59_adam_transformer_encoder_3_multi_head_attention_value_kernel_m:b
Passignvariableop_60_adam_transformer_encoder_3_multi_head_attention_value_bias_m:s
]assignvariableop_61_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_m:i
[assignvariableop_62_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_m:X
Jassignvariableop_63_adam_transformer_encoder_3_layer_normalization_gamma_m:W
Iassignvariableop_64_adam_transformer_encoder_3_layer_normalization_beta_m:Z
Lassignvariableop_65_adam_transformer_encoder_3_layer_normalization_1_gamma_m:Y
Kassignvariableop_66_adam_transformer_encoder_3_layer_normalization_1_beta_m:O
=assignvariableop_67_adam_transformer_encoder_3_dense_kernel_m:I
;assignvariableop_68_adam_transformer_encoder_3_dense_bias_m:Q
?assignvariableop_69_adam_transformer_encoder_3_dense_1_kernel_m:K
=assignvariableop_70_adam_transformer_encoder_3_dense_1_bias_m:h
Rassignvariableop_71_adam_transformer_decoder_3_multi_head_attention_query_kernel_m:b
Passignvariableop_72_adam_transformer_decoder_3_multi_head_attention_query_bias_m:f
Passignvariableop_73_adam_transformer_decoder_3_multi_head_attention_key_kernel_m:`
Nassignvariableop_74_adam_transformer_decoder_3_multi_head_attention_key_bias_m:h
Rassignvariableop_75_adam_transformer_decoder_3_multi_head_attention_value_kernel_m:b
Passignvariableop_76_adam_transformer_decoder_3_multi_head_attention_value_bias_m:s
]assignvariableop_77_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_m:i
[assignvariableop_78_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_m:X
Jassignvariableop_79_adam_transformer_decoder_3_layer_normalization_gamma_m:W
Iassignvariableop_80_adam_transformer_decoder_3_layer_normalization_beta_m:Z
Lassignvariableop_81_adam_transformer_decoder_3_layer_normalization_1_gamma_m:Y
Kassignvariableop_82_adam_transformer_decoder_3_layer_normalization_1_beta_m:O
=assignvariableop_83_adam_transformer_decoder_3_dense_kernel_m:I
;assignvariableop_84_adam_transformer_decoder_3_dense_bias_m:Q
?assignvariableop_85_adam_transformer_decoder_3_dense_1_kernel_m:K
=assignvariableop_86_adam_transformer_decoder_3_dense_1_bias_m:C
1assignvariableop_87_adam_embedding_3_embeddings_v:;
)assignvariableop_88_adam_dense_4_kernel_v:@5
'assignvariableop_89_adam_dense_4_bias_v:@;
)assignvariableop_90_adam_dense_5_kernel_v:@ 5
'assignvariableop_91_adam_dense_5_bias_v: h
Uassignvariableop_92_adam_token_and_position_embedding_2_token_embedding3_embeddings_v:	?j
Xassignvariableop_93_adam_token_and_position_embedding_2_position_embedding3_embeddings_v: h
Rassignvariableop_94_adam_transformer_encoder_3_multi_head_attention_query_kernel_v:b
Passignvariableop_95_adam_transformer_encoder_3_multi_head_attention_query_bias_v:f
Passignvariableop_96_adam_transformer_encoder_3_multi_head_attention_key_kernel_v:`
Nassignvariableop_97_adam_transformer_encoder_3_multi_head_attention_key_bias_v:h
Rassignvariableop_98_adam_transformer_encoder_3_multi_head_attention_value_kernel_v:b
Passignvariableop_99_adam_transformer_encoder_3_multi_head_attention_value_bias_v:t
^assignvariableop_100_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_v:j
\assignvariableop_101_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_v:Y
Kassignvariableop_102_adam_transformer_encoder_3_layer_normalization_gamma_v:X
Jassignvariableop_103_adam_transformer_encoder_3_layer_normalization_beta_v:[
Massignvariableop_104_adam_transformer_encoder_3_layer_normalization_1_gamma_v:Z
Lassignvariableop_105_adam_transformer_encoder_3_layer_normalization_1_beta_v:P
>assignvariableop_106_adam_transformer_encoder_3_dense_kernel_v:J
<assignvariableop_107_adam_transformer_encoder_3_dense_bias_v:R
@assignvariableop_108_adam_transformer_encoder_3_dense_1_kernel_v:L
>assignvariableop_109_adam_transformer_encoder_3_dense_1_bias_v:i
Sassignvariableop_110_adam_transformer_decoder_3_multi_head_attention_query_kernel_v:c
Qassignvariableop_111_adam_transformer_decoder_3_multi_head_attention_query_bias_v:g
Qassignvariableop_112_adam_transformer_decoder_3_multi_head_attention_key_kernel_v:a
Oassignvariableop_113_adam_transformer_decoder_3_multi_head_attention_key_bias_v:i
Sassignvariableop_114_adam_transformer_decoder_3_multi_head_attention_value_kernel_v:c
Qassignvariableop_115_adam_transformer_decoder_3_multi_head_attention_value_bias_v:t
^assignvariableop_116_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_v:j
\assignvariableop_117_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_v:Y
Kassignvariableop_118_adam_transformer_decoder_3_layer_normalization_gamma_v:X
Jassignvariableop_119_adam_transformer_decoder_3_layer_normalization_beta_v:[
Massignvariableop_120_adam_transformer_decoder_3_layer_normalization_1_gamma_v:Z
Lassignvariableop_121_adam_transformer_decoder_3_layer_normalization_1_beta_v:P
>assignvariableop_122_adam_transformer_decoder_3_dense_kernel_v:J
<assignvariableop_123_adam_transformer_decoder_3_dense_bias_v:R
@assignvariableop_124_adam_transformer_decoder_3_dense_1_kernel_v:L
>assignvariableop_125_adam_transformer_decoder_3_dense_1_bias_v:
identity_127??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?2MutableHashTable_table_restore/LookupTableImportV2?=
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?=
value?=B?=?B:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_3_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_4_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_4_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_5_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_5_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpMassignvariableop_5_token_and_position_embedding_2_token_embedding3_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpPassignvariableop_6_token_and_position_embedding_2_position_embedding3_embeddingsIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpJassignvariableop_7_transformer_encoder_3_multi_head_attention_query_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpHassignvariableop_8_transformer_encoder_3_multi_head_attention_query_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpHassignvariableop_9_transformer_encoder_3_multi_head_attention_key_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpGassignvariableop_10_transformer_encoder_3_multi_head_attention_key_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpKassignvariableop_11_transformer_encoder_3_multi_head_attention_value_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpIassignvariableop_12_transformer_encoder_3_multi_head_attention_value_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpVassignvariableop_13_transformer_encoder_3_multi_head_attention_attention_output_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpTassignvariableop_14_transformer_encoder_3_multi_head_attention_attention_output_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpCassignvariableop_15_transformer_encoder_3_layer_normalization_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpBassignvariableop_16_transformer_encoder_3_layer_normalization_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpEassignvariableop_17_transformer_encoder_3_layer_normalization_1_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpDassignvariableop_18_transformer_encoder_3_layer_normalization_1_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_transformer_encoder_3_dense_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_transformer_encoder_3_dense_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp8assignvariableop_21_transformer_encoder_3_dense_1_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_transformer_encoder_3_dense_1_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpKassignvariableop_23_transformer_decoder_3_multi_head_attention_query_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpIassignvariableop_24_transformer_decoder_3_multi_head_attention_query_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_decoder_3_multi_head_attention_key_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpGassignvariableop_26_transformer_decoder_3_multi_head_attention_key_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpKassignvariableop_27_transformer_decoder_3_multi_head_attention_value_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpIassignvariableop_28_transformer_decoder_3_multi_head_attention_value_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpVassignvariableop_29_transformer_decoder_3_multi_head_attention_attention_output_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpTassignvariableop_30_transformer_decoder_3_multi_head_attention_attention_output_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpCassignvariableop_31_transformer_decoder_3_layer_normalization_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpBassignvariableop_32_transformer_decoder_3_layer_normalization_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpEassignvariableop_33_transformer_decoder_3_layer_normalization_1_gammaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpDassignvariableop_34_transformer_decoder_3_layer_normalization_1_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_transformer_decoder_3_dense_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp4assignvariableop_36_transformer_decoder_3_dense_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp8assignvariableop_37_transformer_decoder_3_dense_1_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_transformer_decoder_3_dense_1_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_adam_iterIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_beta_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_2Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_decayIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp&assignvariableop_43_adam_learning_rateIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:44RestoreV2:tensors:45*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_44IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_total_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_count_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_totalIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_countIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp1assignvariableop_48_adam_embedding_3_embeddings_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_4_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_4_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_5_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_5_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpUassignvariableop_53_adam_token_and_position_embedding_2_token_embedding3_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpXassignvariableop_54_adam_token_and_position_embedding_2_position_embedding3_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpRassignvariableop_55_adam_transformer_encoder_3_multi_head_attention_query_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpPassignvariableop_56_adam_transformer_encoder_3_multi_head_attention_query_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpPassignvariableop_57_adam_transformer_encoder_3_multi_head_attention_key_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpNassignvariableop_58_adam_transformer_encoder_3_multi_head_attention_key_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpRassignvariableop_59_adam_transformer_encoder_3_multi_head_attention_value_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpPassignvariableop_60_adam_transformer_encoder_3_multi_head_attention_value_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp]assignvariableop_61_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp[assignvariableop_62_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOpJassignvariableop_63_adam_transformer_encoder_3_layer_normalization_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpIassignvariableop_64_adam_transformer_encoder_3_layer_normalization_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpLassignvariableop_65_adam_transformer_encoder_3_layer_normalization_1_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpKassignvariableop_66_adam_transformer_encoder_3_layer_normalization_1_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp=assignvariableop_67_adam_transformer_encoder_3_dense_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp;assignvariableop_68_adam_transformer_encoder_3_dense_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp?assignvariableop_69_adam_transformer_encoder_3_dense_1_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp=assignvariableop_70_adam_transformer_encoder_3_dense_1_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpRassignvariableop_71_adam_transformer_decoder_3_multi_head_attention_query_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpPassignvariableop_72_adam_transformer_decoder_3_multi_head_attention_query_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOpPassignvariableop_73_adam_transformer_decoder_3_multi_head_attention_key_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOpNassignvariableop_74_adam_transformer_decoder_3_multi_head_attention_key_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOpRassignvariableop_75_adam_transformer_decoder_3_multi_head_attention_value_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOpPassignvariableop_76_adam_transformer_decoder_3_multi_head_attention_value_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp]assignvariableop_77_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp[assignvariableop_78_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOpJassignvariableop_79_adam_transformer_decoder_3_layer_normalization_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOpIassignvariableop_80_adam_transformer_decoder_3_layer_normalization_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOpLassignvariableop_81_adam_transformer_decoder_3_layer_normalization_1_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOpKassignvariableop_82_adam_transformer_decoder_3_layer_normalization_1_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp=assignvariableop_83_adam_transformer_decoder_3_dense_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp;assignvariableop_84_adam_transformer_decoder_3_dense_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp?assignvariableop_85_adam_transformer_decoder_3_dense_1_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp=assignvariableop_86_adam_transformer_decoder_3_dense_1_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp1assignvariableop_87_adam_embedding_3_embeddings_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_dense_4_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adam_dense_4_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_dense_5_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp'assignvariableop_91_adam_dense_5_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOpUassignvariableop_92_adam_token_and_position_embedding_2_token_embedding3_embeddings_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOpXassignvariableop_93_adam_token_and_position_embedding_2_position_embedding3_embeddings_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOpRassignvariableop_94_adam_transformer_encoder_3_multi_head_attention_query_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOpPassignvariableop_95_adam_transformer_encoder_3_multi_head_attention_query_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOpPassignvariableop_96_adam_transformer_encoder_3_multi_head_attention_key_kernel_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOpNassignvariableop_97_adam_transformer_encoder_3_multi_head_attention_key_bias_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0`
Identity_98IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOpRassignvariableop_98_adam_transformer_encoder_3_multi_head_attention_value_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0`
Identity_99IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOpPassignvariableop_99_adam_transformer_encoder_3_multi_head_attention_value_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp^assignvariableop_100_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp\assignvariableop_101_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOpKassignvariableop_102_adam_transformer_encoder_3_layer_normalization_gamma_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOpJassignvariableop_103_adam_transformer_encoder_3_layer_normalization_beta_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOpMassignvariableop_104_adam_transformer_encoder_3_layer_normalization_1_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOpLassignvariableop_105_adam_transformer_encoder_3_layer_normalization_1_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp>assignvariableop_106_adam_transformer_encoder_3_dense_kernel_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp<assignvariableop_107_adam_transformer_encoder_3_dense_bias_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp@assignvariableop_108_adam_transformer_encoder_3_dense_1_kernel_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp>assignvariableop_109_adam_transformer_encoder_3_dense_1_bias_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOpSassignvariableop_110_adam_transformer_decoder_3_multi_head_attention_query_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOpQassignvariableop_111_adam_transformer_decoder_3_multi_head_attention_query_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOpQassignvariableop_112_adam_transformer_decoder_3_multi_head_attention_key_kernel_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOpOassignvariableop_113_adam_transformer_decoder_3_multi_head_attention_key_bias_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOpSassignvariableop_114_adam_transformer_decoder_3_multi_head_attention_value_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOpQassignvariableop_115_adam_transformer_decoder_3_multi_head_attention_value_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp^assignvariableop_116_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp\assignvariableop_117_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOpKassignvariableop_118_adam_transformer_decoder_3_layer_normalization_gamma_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOpJassignvariableop_119_adam_transformer_decoder_3_layer_normalization_beta_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOpMassignvariableop_120_adam_transformer_decoder_3_layer_normalization_1_gamma_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOpLassignvariableop_121_adam_transformer_decoder_3_layer_normalization_1_beta_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp>assignvariableop_122_adam_transformer_decoder_3_dense_kernel_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp<assignvariableop_123_adam_transformer_decoder_3_dense_bias_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOp@assignvariableop_124_adam_transformer_decoder_3_dense_1_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp>assignvariableop_125_adam_transformer_decoder_3_dense_1_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_126Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_993^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_127IdentityIdentity_126:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_993^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "%
identity_127Identity_127:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_125AssignVariableOp_1252*
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
AssignVariableOp_99AssignVariableOp_992h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)-%
#
_class
loc:@MutableHashTable
?!
?
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_335768

inputs	;
(token_embedding3_embedding_lookup_335744:	?=
+position_embedding3_readvariableop_resource: 
identity??"position_embedding3/ReadVariableOp?!token_embedding3/embedding_lookup?
!token_embedding3/embedding_lookupResourceGather(token_embedding3_embedding_lookup_335744inputs*
Tindices0	*;
_class1
/-loc:@token_embedding3/embedding_lookup/335744*+
_output_shapes
:????????? *
dtype0?
*token_embedding3/embedding_lookup/IdentityIdentity*token_embedding3/embedding_lookup:output:0*
T0*;
_class1
/-loc:@token_embedding3/embedding_lookup/335744*+
_output_shapes
:????????? ?
,token_embedding3/embedding_lookup/Identity_1Identity3token_embedding3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ~
position_embedding3/ShapeShape5token_embedding3/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:z
'position_embedding3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
)position_embedding3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????s
)position_embedding3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!position_embedding3/strided_sliceStridedSlice"position_embedding3/Shape:output:00position_embedding3/strided_slice/stack:output:02position_embedding3/strided_slice/stack_1:output:02position_embedding3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"position_embedding3/ReadVariableOpReadVariableOp+position_embedding3_readvariableop_resource*
_output_shapes

: *
dtype0[
position_embedding3/ConstConst*
_output_shapes
: *
dtype0*
value	B : ]
position_embedding3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :m
+position_embedding3/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
)position_embedding3/strided_slice_1/stackPack"position_embedding3/Const:output:04position_embedding3/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:o
-position_embedding3/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
+position_embedding3/strided_slice_1/stack_1Pack*position_embedding3/strided_slice:output:06position_embedding3/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:o
-position_embedding3/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
+position_embedding3/strided_slice_1/stack_2Pack$position_embedding3/Const_1:output:06position_embedding3/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
#position_embedding3/strided_slice_1StridedSlice*position_embedding3/ReadVariableOp:value:02position_embedding3/strided_slice_1/stack:output:04position_embedding3/strided_slice_1/stack_1:output:04position_embedding3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
position_embedding3/BroadcastToBroadcastTo,position_embedding3/strided_slice_1:output:0"position_embedding3/Shape:output:0*
T0*+
_output_shapes
:????????? ?
addAddV25token_embedding3/embedding_lookup/Identity_1:output:0(position_embedding3/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp#^position_embedding3/ReadVariableOp"^token_embedding3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2H
"position_embedding3/ReadVariableOp"position_embedding3/ReadVariableOp2F
!token_embedding3/embedding_lookup!token_embedding3/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
(__inference_dense_5_layer_call_fn_336627

inputs
unknown:@ 
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
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_333308o
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
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
6__inference_transformer_decoder_3_layer_call_fn_336220
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
	unknown_9:

unknown_10:

unknown_11:

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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333658s
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
?
r
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_336598

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
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333245
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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
??
?
C__inference_model_2_layer_call_and_return_conditional_losses_334173

inputs
inputs_1U
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	8
%token_and_position_embedding_2_334086:	?7
%token_and_position_embedding_2_334088: $
embedding_3_334091:2
transformer_encoder_3_334095:.
transformer_encoder_3_334097:2
transformer_encoder_3_334099:.
transformer_encoder_3_334101:2
transformer_encoder_3_334103:.
transformer_encoder_3_334105:2
transformer_encoder_3_334107:*
transformer_encoder_3_334109:*
transformer_encoder_3_334111:*
transformer_encoder_3_334113:.
transformer_encoder_3_334115:*
transformer_encoder_3_334117:.
transformer_encoder_3_334119:*
transformer_encoder_3_334121:*
transformer_encoder_3_334123:*
transformer_encoder_3_334125:2
transformer_decoder_3_334128:.
transformer_decoder_3_334130:2
transformer_decoder_3_334132:.
transformer_decoder_3_334134:2
transformer_decoder_3_334136:.
transformer_decoder_3_334138:2
transformer_decoder_3_334140:*
transformer_decoder_3_334142:*
transformer_decoder_3_334144:*
transformer_decoder_3_334146:.
transformer_decoder_3_334148:*
transformer_decoder_3_334150:.
transformer_decoder_3_334152:*
transformer_decoder_3_334154:*
transformer_decoder_3_334156:*
transformer_decoder_3_334158: 
dense_4_334162:@
dense_4_334164:@ 
dense_5_334167:@ 
dense_5_334169: 
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_3/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?6token_and_position_embedding_2/StatefulPartitionedCall?-transformer_decoder_3/StatefulPartitionedCall?-transformer_encoder_3/StatefulPartitionedCall{
text_vectorization/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0%token_and_position_embedding_2_334086%token_and_position_embedding_2_334088*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_332883?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_3_334091*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_332899?
add_2/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_332909?
-transformer_encoder_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_encoder_3_334095transformer_encoder_3_334097transformer_encoder_3_334099transformer_encoder_3_334101transformer_encoder_3_334103transformer_encoder_3_334105transformer_encoder_3_334107transformer_encoder_3_334109transformer_encoder_3_334111transformer_encoder_3_334113transformer_encoder_3_334115transformer_encoder_3_334117transformer_encoder_3_334119transformer_encoder_3_334121transformer_encoder_3_334123transformer_encoder_3_334125*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333881?
-transformer_decoder_3/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_3/StatefulPartitionedCall:output:0transformer_decoder_3_334128transformer_decoder_3_334130transformer_decoder_3_334132transformer_decoder_3_334134transformer_decoder_3_334136transformer_decoder_3_334138transformer_decoder_3_334140transformer_decoder_3_334142transformer_decoder_3_334144transformer_decoder_3_334146transformer_decoder_3_334148transformer_decoder_3_334150transformer_decoder_3_334152transformer_decoder_3_334154transformer_decoder_3_334156transformer_decoder_3_334158*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333658?
*global_average_pooling1d_2/PartitionedCallPartitionedCall6transformer_decoder_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_332798?
dense_4/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_4_334162dense_4_334164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_333291?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_334167dense_5_334169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_333308w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV27^token_and_position_embedding_2/StatefulPartitionedCall.^transformer_decoder_3/StatefulPartitionedCall.^transformer_encoder_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2^
-transformer_decoder_3/StatefulPartitionedCall-transformer_decoder_3/StatefulPartitionedCall2^
-transformer_encoder_3/StatefulPartitionedCall-transformer_encoder_3/StatefulPartitionedCall:O K
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
?
?

$__inference_signature_wrapper_334728

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24:

unknown_25: 

unknown_26:

unknown_27: 

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:@

unknown_39:@

unknown_40:@ 

unknown_41: 
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
unknown_40
unknown_41*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *I
_read_only_resource_inputs+
)'	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_332788o
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
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namePhrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
Token_role:
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
k
A__inference_add_2_layer_call_and_return_conditional_losses_332909

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
?
+
__inference_<lambda>_336711
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
,__inference_embedding_3_layer_call_fn_335775

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_332899s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:????????? : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_332883

inputs	;
(token_embedding3_embedding_lookup_332859:	?=
+position_embedding3_readvariableop_resource: 
identity??"position_embedding3/ReadVariableOp?!token_embedding3/embedding_lookup?
!token_embedding3/embedding_lookupResourceGather(token_embedding3_embedding_lookup_332859inputs*
Tindices0	*;
_class1
/-loc:@token_embedding3/embedding_lookup/332859*+
_output_shapes
:????????? *
dtype0?
*token_embedding3/embedding_lookup/IdentityIdentity*token_embedding3/embedding_lookup:output:0*
T0*;
_class1
/-loc:@token_embedding3/embedding_lookup/332859*+
_output_shapes
:????????? ?
,token_embedding3/embedding_lookup/Identity_1Identity3token_embedding3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ~
position_embedding3/ShapeShape5token_embedding3/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:z
'position_embedding3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
)position_embedding3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????s
)position_embedding3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!position_embedding3/strided_sliceStridedSlice"position_embedding3/Shape:output:00position_embedding3/strided_slice/stack:output:02position_embedding3/strided_slice/stack_1:output:02position_embedding3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"position_embedding3/ReadVariableOpReadVariableOp+position_embedding3_readvariableop_resource*
_output_shapes

: *
dtype0[
position_embedding3/ConstConst*
_output_shapes
: *
dtype0*
value	B : ]
position_embedding3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :m
+position_embedding3/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
)position_embedding3/strided_slice_1/stackPack"position_embedding3/Const:output:04position_embedding3/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:o
-position_embedding3/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
+position_embedding3/strided_slice_1/stack_1Pack*position_embedding3/strided_slice:output:06position_embedding3/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:o
-position_embedding3/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
+position_embedding3/strided_slice_1/stack_2Pack$position_embedding3/Const_1:output:06position_embedding3/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
#position_embedding3/strided_slice_1StridedSlice*position_embedding3/ReadVariableOp:value:02position_embedding3/strided_slice_1/stack:output:04position_embedding3/strided_slice_1/stack_1:output:04position_embedding3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
position_embedding3/BroadcastToBroadcastTo,position_embedding3/strided_slice_1:output:0"position_embedding3/Shape:output:0*
T0*+
_output_shapes
:????????? ?
addAddV25token_embedding3/embedding_lookup/Identity_1:output:0(position_embedding3/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? Z
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:????????? ?
NoOpNoOp#^position_embedding3/ReadVariableOp"^token_embedding3/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2H
"position_embedding3/ReadVariableOp"position_embedding3/ReadVariableOp2F
!token_embedding3/embedding_lookup!token_embedding3/embedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_4_layer_call_and_return_conditional_losses_333291

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
??
?3
C__inference_model_2_layer_call_and_return_conditional_losses_335301
inputs_0
inputs_1U
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	Z
Gtoken_and_position_embedding_2_token_embedding3_embedding_lookup_334962:	?\
Jtoken_and_position_embedding_2_position_embedding3_readvariableop_resource: 5
#embedding_3_embedding_lookup_334986:l
Vtransformer_encoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:^
Ltransformer_encoder_3_multi_head_attention_query_add_readvariableop_resource:j
Ttransformer_encoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:\
Jtransformer_encoder_3_multi_head_attention_key_add_readvariableop_resource:l
Vtransformer_encoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:^
Ltransformer_encoder_3_multi_head_attention_value_add_readvariableop_resource:w
atransformer_encoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:e
Wtransformer_encoder_3_multi_head_attention_attention_output_add_readvariableop_resource:]
Otransformer_encoder_3_layer_normalization_batchnorm_mul_readvariableop_resource:Y
Ktransformer_encoder_3_layer_normalization_batchnorm_readvariableop_resource:O
=transformer_encoder_3_dense_tensordot_readvariableop_resource:I
;transformer_encoder_3_dense_biasadd_readvariableop_resource:Q
?transformer_encoder_3_dense_1_tensordot_readvariableop_resource:K
=transformer_encoder_3_dense_1_biasadd_readvariableop_resource:_
Qtransformer_encoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:[
Mtransformer_encoder_3_layer_normalization_1_batchnorm_readvariableop_resource:l
Vtransformer_decoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:^
Ltransformer_decoder_3_multi_head_attention_query_add_readvariableop_resource:j
Ttransformer_decoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:\
Jtransformer_decoder_3_multi_head_attention_key_add_readvariableop_resource:l
Vtransformer_decoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:^
Ltransformer_decoder_3_multi_head_attention_value_add_readvariableop_resource:w
atransformer_decoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:e
Wtransformer_decoder_3_multi_head_attention_attention_output_add_readvariableop_resource:]
Otransformer_decoder_3_layer_normalization_batchnorm_mul_readvariableop_resource:Y
Ktransformer_decoder_3_layer_normalization_batchnorm_readvariableop_resource:O
=transformer_decoder_3_dense_tensordot_readvariableop_resource:I
;transformer_decoder_3_dense_biasadd_readvariableop_resource:Q
?transformer_decoder_3_dense_1_tensordot_readvariableop_resource:K
=transformer_decoder_3_dense_1_biasadd_readvariableop_resource:_
Qtransformer_decoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:[
Mtransformer_decoder_3_layer_normalization_1_batchnorm_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@ 5
'dense_5_biasadd_readvariableop_resource: 
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?embedding_3/embedding_lookup?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?Atoken_and_position_embedding_2/position_embedding3/ReadVariableOp?@token_and_position_embedding_2/token_embedding3/embedding_lookup?2transformer_decoder_3/dense/BiasAdd/ReadVariableOp?4transformer_decoder_3/dense/Tensordot/ReadVariableOp?4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp?6transformer_decoder_3/dense_1/Tensordot/ReadVariableOp?Btransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp?Ftransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp?Dtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp?Htransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp?Ntransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp?Xtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_decoder_3/multi_head_attention/key/add/ReadVariableOp?Ktransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Ctransformer_decoder_3/multi_head_attention/query/add/ReadVariableOp?Mtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Ctransformer_decoder_3/multi_head_attention/value/add/ReadVariableOp?Mtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp?2transformer_encoder_3/dense/BiasAdd/ReadVariableOp?4transformer_encoder_3/dense/Tensordot/ReadVariableOp?4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp?6transformer_encoder_3/dense_1/Tensordot/ReadVariableOp?Btransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp?Ftransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp?Dtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp?Htransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp?Ntransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp?Xtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp?Atransformer_encoder_3/multi_head_attention/key/add/ReadVariableOp?Ktransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp?Ctransformer_encoder_3/multi_head_attention/query/add/ReadVariableOp?Mtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp?Ctransformer_encoder_3/multi_head_attention/value/add/ReadVariableOp?Mtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp}
text_vectorization/SqueezeSqueezeinputs_0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
@token_and_position_embedding_2/token_embedding3/embedding_lookupResourceGatherGtoken_and_position_embedding_2_token_embedding3_embedding_lookup_334962?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*Z
_classP
NLloc:@token_and_position_embedding_2/token_embedding3/embedding_lookup/334962*+
_output_shapes
:????????? *
dtype0?
Itoken_and_position_embedding_2/token_embedding3/embedding_lookup/IdentityIdentityItoken_and_position_embedding_2/token_embedding3/embedding_lookup:output:0*
T0*Z
_classP
NLloc:@token_and_position_embedding_2/token_embedding3/embedding_lookup/334962*+
_output_shapes
:????????? ?
Ktoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1IdentityRtoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
8token_and_position_embedding_2/position_embedding3/ShapeShapeTtoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:?
Ftoken_and_position_embedding_2/position_embedding3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Htoken_and_position_embedding_2/position_embedding3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
??????????
Htoken_and_position_embedding_2/position_embedding3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
@token_and_position_embedding_2/position_embedding3/strided_sliceStridedSliceAtoken_and_position_embedding_2/position_embedding3/Shape:output:0Otoken_and_position_embedding_2/position_embedding3/strided_slice/stack:output:0Qtoken_and_position_embedding_2/position_embedding3/strided_slice/stack_1:output:0Qtoken_and_position_embedding_2/position_embedding3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Atoken_and_position_embedding_2/position_embedding3/ReadVariableOpReadVariableOpJtoken_and_position_embedding_2_position_embedding3_readvariableop_resource*
_output_shapes

: *
dtype0z
8token_and_position_embedding_2/position_embedding3/ConstConst*
_output_shapes
: *
dtype0*
value	B : |
:token_and_position_embedding_2/position_embedding3/Const_1Const*
_output_shapes
: *
dtype0*
value	B :?
Jtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Htoken_and_position_embedding_2/position_embedding3/strided_slice_1/stackPackAtoken_and_position_embedding_2/position_embedding3/Const:output:0Stoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:?
Ltoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ?
Jtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1PackItoken_and_position_embedding_2/position_embedding3/strided_slice:output:0Utoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:?
Ltoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2/1Const*
_output_shapes
: *
dtype0*
value	B :?
Jtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2PackCtoken_and_position_embedding_2/position_embedding3/Const_1:output:0Utoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2/1:output:0*
N*
T0*
_output_shapes
:?
Btoken_and_position_embedding_2/position_embedding3/strided_slice_1StridedSliceItoken_and_position_embedding_2/position_embedding3/ReadVariableOp:value:0Qtoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack:output:0Stoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_1:output:0Stoken_and_position_embedding_2/position_embedding3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask?
>token_and_position_embedding_2/position_embedding3/BroadcastToBroadcastToKtoken_and_position_embedding_2/position_embedding3/strided_slice_1:output:0Atoken_and_position_embedding_2/position_embedding3/Shape:output:0*
T0*+
_output_shapes
:????????? ?
"token_and_position_embedding_2/addAddV2Ttoken_and_position_embedding_2/token_embedding3/embedding_lookup/Identity_1:output:0Gtoken_and_position_embedding_2/position_embedding3/BroadcastTo:output:0*
T0*+
_output_shapes
:????????? c
embedding_3/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
embedding_3/embedding_lookupResourceGather#embedding_3_embedding_lookup_334986embedding_3/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding_3/embedding_lookup/334986*+
_output_shapes
:????????? *
dtype0?
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_3/embedding_lookup/334986*+
_output_shapes
:????????? ?
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? ?
	add_2/addAddV2&token_and_position_embedding_2/add:z:00embedding_3/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:????????? ?
Mtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_encoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_encoder_3/multi_head_attention/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_encoder_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpLtransformer_encoder_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_encoder_3/multi_head_attention/query/addAddV2Gtransformer_encoder_3/multi_head_attention/query/einsum/Einsum:output:0Ktransformer_encoder_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ktransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_encoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
<transformer_encoder_3/multi_head_attention/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Atransformer_encoder_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpJtransformer_encoder_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
2transformer_encoder_3/multi_head_attention/key/addAddV2Etransformer_encoder_3/multi_head_attention/key/einsum/Einsum:output:0Itransformer_encoder_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Mtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_encoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_encoder_3/multi_head_attention/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_encoder_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpLtransformer_encoder_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_encoder_3/multi_head_attention/value/addAddV2Gtransformer_encoder_3/multi_head_attention/value/einsum/Einsum:output:0Ktransformer_encoder_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? u
0transformer_encoder_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
.transformer_encoder_3/multi_head_attention/MulMul8transformer_encoder_3/multi_head_attention/query/add:z:09transformer_encoder_3/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
8transformer_encoder_3/multi_head_attention/einsum/EinsumEinsum6transformer_encoder_3/multi_head_attention/key/add:z:02transformer_encoder_3/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
:transformer_encoder_3/multi_head_attention/softmax/SoftmaxSoftmaxAtransformer_encoder_3/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:?????????  ?
=transformer_encoder_3/multi_head_attention/dropout_2/IdentityIdentityDtransformer_encoder_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
:transformer_encoder_3/multi_head_attention/einsum_1/EinsumEinsumFtransformer_encoder_3/multi_head_attention/dropout_2/Identity:output:08transformer_encoder_3/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Xtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_encoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Itransformer_encoder_3/multi_head_attention/attention_output/einsum/EinsumEinsumCtransformer_encoder_3/multi_head_attention/einsum_1/Einsum:output:0`transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Ntransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWtransformer_encoder_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
?transformer_encoder_3/multi_head_attention/attention_output/addAddV2Rtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum:output:0Vtransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
&transformer_encoder_3/dropout/IdentityIdentityCtransformer_encoder_3/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_3/addAddV2add_2/add:z:0/transformer_encoder_3/dropout/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Htransformer_encoder_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_encoder_3/layer_normalization/moments/meanMeantransformer_encoder_3/add:z:0Qtransformer_encoder_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
>transformer_encoder_3/layer_normalization/moments/StopGradientStopGradient?transformer_encoder_3/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_encoder_3/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_3/add:z:0Gtransformer_encoder_3/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ltransformer_encoder_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_encoder_3/layer_normalization/moments/varianceMeanGtransformer_encoder_3/layer_normalization/moments/SquaredDifference:z:0Utransformer_encoder_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(~
9transformer_encoder_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
7transformer_encoder_3/layer_normalization/batchnorm/addAddV2Ctransformer_encoder_3/layer_normalization/moments/variance:output:0Btransformer_encoder_3/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/RsqrtRsqrt;transformer_encoder_3/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_encoder_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_encoder_3/layer_normalization/batchnorm/mulMul=transformer_encoder_3/layer_normalization/batchnorm/Rsqrt:y:0Ntransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/mul_1Multransformer_encoder_3/add:z:0;transformer_encoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/mul_2Mul?transformer_encoder_3/layer_normalization/moments/mean:output:0;transformer_encoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Btransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKtransformer_encoder_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_encoder_3/layer_normalization/batchnorm/subSubJtransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp:value:0=transformer_encoder_3/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_encoder_3/layer_normalization/batchnorm/add_1AddV2=transformer_encoder_3/layer_normalization/batchnorm/mul_1:z:0;transformer_encoder_3/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
4transformer_encoder_3/dense/Tensordot/ReadVariableOpReadVariableOp=transformer_encoder_3_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0t
*transformer_encoder_3/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:{
*transformer_encoder_3/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
+transformer_encoder_3/dense/Tensordot/ShapeShape=transformer_encoder_3/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
3transformer_encoder_3/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_encoder_3/dense/Tensordot/GatherV2GatherV24transformer_encoder_3/dense/Tensordot/Shape:output:03transformer_encoder_3/dense/Tensordot/free:output:0<transformer_encoder_3/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
5transformer_encoder_3/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_encoder_3/dense/Tensordot/GatherV2_1GatherV24transformer_encoder_3/dense/Tensordot/Shape:output:03transformer_encoder_3/dense/Tensordot/axes:output:0>transformer_encoder_3/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
+transformer_encoder_3/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*transformer_encoder_3/dense/Tensordot/ProdProd7transformer_encoder_3/dense/Tensordot/GatherV2:output:04transformer_encoder_3/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: w
-transformer_encoder_3/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_encoder_3/dense/Tensordot/Prod_1Prod9transformer_encoder_3/dense/Tensordot/GatherV2_1:output:06transformer_encoder_3/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: s
1transformer_encoder_3/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,transformer_encoder_3/dense/Tensordot/concatConcatV23transformer_encoder_3/dense/Tensordot/free:output:03transformer_encoder_3/dense/Tensordot/axes:output:0:transformer_encoder_3/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+transformer_encoder_3/dense/Tensordot/stackPack3transformer_encoder_3/dense/Tensordot/Prod:output:05transformer_encoder_3/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
/transformer_encoder_3/dense/Tensordot/transpose	Transpose=transformer_encoder_3/layer_normalization/batchnorm/add_1:z:05transformer_encoder_3/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_encoder_3/dense/Tensordot/ReshapeReshape3transformer_encoder_3/dense/Tensordot/transpose:y:04transformer_encoder_3/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
,transformer_encoder_3/dense/Tensordot/MatMulMatMul6transformer_encoder_3/dense/Tensordot/Reshape:output:0<transformer_encoder_3/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
-transformer_encoder_3/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:u
3transformer_encoder_3/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_encoder_3/dense/Tensordot/concat_1ConcatV27transformer_encoder_3/dense/Tensordot/GatherV2:output:06transformer_encoder_3/dense/Tensordot/Const_2:output:0<transformer_encoder_3/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
%transformer_encoder_3/dense/TensordotReshape6transformer_encoder_3/dense/Tensordot/MatMul:product:07transformer_encoder_3/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
2transformer_encoder_3/dense/BiasAdd/ReadVariableOpReadVariableOp;transformer_encoder_3_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#transformer_encoder_3/dense/BiasAddBiasAdd.transformer_encoder_3/dense/Tensordot:output:0:transformer_encoder_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
 transformer_encoder_3/dense/ReluRelu,transformer_encoder_3/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
6transformer_encoder_3/dense_1/Tensordot/ReadVariableOpReadVariableOp?transformer_encoder_3_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0v
,transformer_encoder_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,transformer_encoder_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
-transformer_encoder_3/dense_1/Tensordot/ShapeShape.transformer_encoder_3/dense/Relu:activations:0*
T0*
_output_shapes
:w
5transformer_encoder_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_encoder_3/dense_1/Tensordot/GatherV2GatherV26transformer_encoder_3/dense_1/Tensordot/Shape:output:05transformer_encoder_3/dense_1/Tensordot/free:output:0>transformer_encoder_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7transformer_encoder_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2transformer_encoder_3/dense_1/Tensordot/GatherV2_1GatherV26transformer_encoder_3/dense_1/Tensordot/Shape:output:05transformer_encoder_3/dense_1/Tensordot/axes:output:0@transformer_encoder_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-transformer_encoder_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_encoder_3/dense_1/Tensordot/ProdProd9transformer_encoder_3/dense_1/Tensordot/GatherV2:output:06transformer_encoder_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/transformer_encoder_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
.transformer_encoder_3/dense_1/Tensordot/Prod_1Prod;transformer_encoder_3/dense_1/Tensordot/GatherV2_1:output:08transformer_encoder_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3transformer_encoder_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_encoder_3/dense_1/Tensordot/concatConcatV25transformer_encoder_3/dense_1/Tensordot/free:output:05transformer_encoder_3/dense_1/Tensordot/axes:output:0<transformer_encoder_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
-transformer_encoder_3/dense_1/Tensordot/stackPack5transformer_encoder_3/dense_1/Tensordot/Prod:output:07transformer_encoder_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
1transformer_encoder_3/dense_1/Tensordot/transpose	Transpose.transformer_encoder_3/dense/Relu:activations:07transformer_encoder_3/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
/transformer_encoder_3/dense_1/Tensordot/ReshapeReshape5transformer_encoder_3/dense_1/Tensordot/transpose:y:06transformer_encoder_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
.transformer_encoder_3/dense_1/Tensordot/MatMulMatMul8transformer_encoder_3/dense_1/Tensordot/Reshape:output:0>transformer_encoder_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y
/transformer_encoder_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w
5transformer_encoder_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_encoder_3/dense_1/Tensordot/concat_1ConcatV29transformer_encoder_3/dense_1/Tensordot/GatherV2:output:08transformer_encoder_3/dense_1/Tensordot/Const_2:output:0>transformer_encoder_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
'transformer_encoder_3/dense_1/TensordotReshape8transformer_encoder_3/dense_1/Tensordot/MatMul:product:09transformer_encoder_3/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp=transformer_encoder_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%transformer_encoder_3/dense_1/BiasAddBiasAdd0transformer_encoder_3/dense_1/Tensordot:output:0<transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
(transformer_encoder_3/dropout_1/IdentityIdentity.transformer_encoder_3/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
transformer_encoder_3/add_1AddV2=transformer_encoder_3/layer_normalization/batchnorm/add_1:z:01transformer_encoder_3/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Jtransformer_encoder_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_encoder_3/layer_normalization_1/moments/meanMeantransformer_encoder_3/add_1:z:0Stransformer_encoder_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
@transformer_encoder_3/layer_normalization_1/moments/StopGradientStopGradientAtransformer_encoder_3/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Etransformer_encoder_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_encoder_3/add_1:z:0Itransformer_encoder_3/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ntransformer_encoder_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
<transformer_encoder_3/layer_normalization_1/moments/varianceMeanItransformer_encoder_3/layer_normalization_1/moments/SquaredDifference:z:0Wtransformer_encoder_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
;transformer_encoder_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
9transformer_encoder_3/layer_normalization_1/batchnorm/addAddV2Etransformer_encoder_3/layer_normalization_1/moments/variance:output:0Dtransformer_encoder_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/RsqrtRsqrt=transformer_encoder_3/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_encoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_encoder_3/layer_normalization_1/batchnorm/mulMul?transformer_encoder_3/layer_normalization_1/batchnorm/Rsqrt:y:0Ptransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/mul_1Multransformer_encoder_3/add_1:z:0=transformer_encoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/mul_2MulAtransformer_encoder_3/layer_normalization_1/moments/mean:output:0=transformer_encoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMtransformer_encoder_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_encoder_3/layer_normalization_1/batchnorm/subSubLtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0?transformer_encoder_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_encoder_3/layer_normalization_1/batchnorm/add_1AddV2?transformer_encoder_3/layer_normalization_1/batchnorm/mul_1:z:0=transformer_encoder_3/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_3/ShapeShape?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:s
)transformer_decoder_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+transformer_decoder_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+transformer_decoder_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#transformer_decoder_3/strided_sliceStridedSlice$transformer_decoder_3/Shape:output:02transformer_decoder_3/strided_slice/stack:output:04transformer_decoder_3/strided_slice/stack_1:output:04transformer_decoder_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%transformer_decoder_3/strided_slice_1StridedSlice$transformer_decoder_3/Shape:output:04transformer_decoder_3/strided_slice_1/stack:output:06transformer_decoder_3/strided_slice_1/stack_1:output:06transformer_decoder_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!transformer_decoder_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : c
!transformer_decoder_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_3/rangeRange*transformer_decoder_3/range/start:output:0.transformer_decoder_3/strided_slice_1:output:0*transformer_decoder_3/range/delta:output:0*
_output_shapes
: |
+transformer_decoder_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        ~
-transformer_decoder_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        ~
-transformer_decoder_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
%transformer_decoder_3/strided_slice_2StridedSlice$transformer_decoder_3/range:output:04transformer_decoder_3/strided_slice_2/stack:output:06transformer_decoder_3/strided_slice_2/stack_1:output:06transformer_decoder_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

: *

begin_mask*
end_mask*
new_axis_maske
#transformer_decoder_3/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : e
#transformer_decoder_3/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :?
transformer_decoder_3/range_1Range,transformer_decoder_3/range_1/start:output:0.transformer_decoder_3/strided_slice_1:output:0,transformer_decoder_3/range_1/delta:output:0*
_output_shapes
: ?
"transformer_decoder_3/GreaterEqualGreaterEqual.transformer_decoder_3/strided_slice_2:output:0&transformer_decoder_3/range_1:output:0*
T0*
_output_shapes

:  ?
transformer_decoder_3/CastCast&transformer_decoder_3/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes

:  u
+transformer_decoder_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%transformer_decoder_3/strided_slice_3StridedSlice$transformer_decoder_3/Shape:output:04transformer_decoder_3/strided_slice_3/stack:output:06transformer_decoder_3/strided_slice_3/stack_1:output:06transformer_decoder_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
+transformer_decoder_3/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-transformer_decoder_3/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%transformer_decoder_3/strided_slice_4StridedSlice$transformer_decoder_3/Shape:output:04transformer_decoder_3/strided_slice_4/stack:output:06transformer_decoder_3/strided_slice_4/stack_1:output:06transformer_decoder_3/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%transformer_decoder_3/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :?
#transformer_decoder_3/Reshape/shapePack.transformer_decoder_3/Reshape/shape/0:output:0.transformer_decoder_3/strided_slice_3:output:0.transformer_decoder_3/strided_slice_4:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_3/ReshapeReshapetransformer_decoder_3/Cast:y:0,transformer_decoder_3/Reshape/shape:output:0*
T0*"
_output_shapes
:  o
$transformer_decoder_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 transformer_decoder_3/ExpandDims
ExpandDims,transformer_decoder_3/strided_slice:output:0-transformer_decoder_3/ExpandDims/dim:output:0*
T0*
_output_shapes
:l
transformer_decoder_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      c
!transformer_decoder_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
transformer_decoder_3/concatConcatV2)transformer_decoder_3/ExpandDims:output:0$transformer_decoder_3/Const:output:0*transformer_decoder_3/concat/axis:output:0*
N*
T0*
_output_shapes
:?
transformer_decoder_3/TileTile&transformer_decoder_3/Reshape:output:0%transformer_decoder_3/concat:output:0*
T0*+
_output_shapes
:?????????  ?
Mtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_decoder_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_decoder_3/multi_head_attention/query/einsum/EinsumEinsum?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0Utransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_decoder_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpLtransformer_decoder_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_decoder_3/multi_head_attention/query/addAddV2Gtransformer_decoder_3/multi_head_attention/query/einsum/Einsum:output:0Ktransformer_decoder_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Ktransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_decoder_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
<transformer_decoder_3/multi_head_attention/key/einsum/EinsumEinsum?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0Stransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Atransformer_decoder_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpJtransformer_decoder_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:*
dtype0?
2transformer_decoder_3/multi_head_attention/key/addAddV2Etransformer_decoder_3/multi_head_attention/key/einsum/Einsum:output:0Itransformer_decoder_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
Mtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_decoder_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
>transformer_decoder_3/multi_head_attention/value/einsum/EinsumEinsum?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0Utransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:????????? *
equationabc,cde->abde?
Ctransformer_decoder_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpLtransformer_decoder_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:*
dtype0?
4transformer_decoder_3/multi_head_attention/value/addAddV2Gtransformer_decoder_3/multi_head_attention/value/einsum/Einsum:output:0Ktransformer_decoder_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? u
0transformer_decoder_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
.transformer_decoder_3/multi_head_attention/MulMul8transformer_decoder_3/multi_head_attention/query/add:z:09transformer_decoder_3/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:????????? ?
8transformer_decoder_3/multi_head_attention/einsum/EinsumEinsum6transformer_decoder_3/multi_head_attention/key/add:z:02transformer_decoder_3/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:?????????  *
equationaecd,abcd->acbe?
9transformer_decoder_3/multi_head_attention/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5transformer_decoder_3/multi_head_attention/ExpandDims
ExpandDims#transformer_decoder_3/Tile:output:0Btransformer_decoder_3/multi_head_attention/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????  ?
7transformer_decoder_3/multi_head_attention/softmax/CastCast>transformer_decoder_3/multi_head_attention/ExpandDims:output:0*

DstT0*

SrcT0*/
_output_shapes
:?????????  }
8transformer_decoder_3/multi_head_attention/softmax/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
6transformer_decoder_3/multi_head_attention/softmax/subSubAtransformer_decoder_3/multi_head_attention/softmax/sub/x:output:0;transformer_decoder_3/multi_head_attention/softmax/Cast:y:0*
T0*/
_output_shapes
:?????????  }
8transformer_decoder_3/multi_head_attention/softmax/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *(kn??
6transformer_decoder_3/multi_head_attention/softmax/mulMul:transformer_decoder_3/multi_head_attention/softmax/sub:z:0Atransformer_decoder_3/multi_head_attention/softmax/mul/y:output:0*
T0*/
_output_shapes
:?????????  ?
6transformer_decoder_3/multi_head_attention/softmax/addAddV2Atransformer_decoder_3/multi_head_attention/einsum/Einsum:output:0:transformer_decoder_3/multi_head_attention/softmax/mul:z:0*
T0*/
_output_shapes
:?????????  ?
:transformer_decoder_3/multi_head_attention/softmax/SoftmaxSoftmax:transformer_decoder_3/multi_head_attention/softmax/add:z:0*
T0*/
_output_shapes
:?????????  ?
=transformer_decoder_3/multi_head_attention/dropout_2/IdentityIdentityDtransformer_decoder_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:?????????  ?
:transformer_decoder_3/multi_head_attention/einsum_1/EinsumEinsumFtransformer_decoder_3/multi_head_attention/dropout_2/Identity:output:08transformer_decoder_3/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:????????? *
equationacbe,aecd->abcd?
Xtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_decoder_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:*
dtype0?
Itransformer_decoder_3/multi_head_attention/attention_output/einsum/EinsumEinsumCtransformer_decoder_3/multi_head_attention/einsum_1/Einsum:output:0`transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:????????? *
equationabcd,cde->abe?
Ntransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWtransformer_decoder_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:*
dtype0?
?transformer_decoder_3/multi_head_attention/attention_output/addAddV2Rtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum:output:0Vtransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
&transformer_decoder_3/dropout/IdentityIdentityCtransformer_decoder_3/multi_head_attention/attention_output/add:z:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_3/addAddV2/transformer_decoder_3/dropout/Identity:output:0?transformer_encoder_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_decoder_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
6transformer_decoder_3/layer_normalization/moments/meanMeantransformer_decoder_3/add:z:0Qtransformer_decoder_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
>transformer_decoder_3/layer_normalization/moments/StopGradientStopGradient?transformer_decoder_3/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Ctransformer_decoder_3/layer_normalization/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_3/add:z:0Gtransformer_decoder_3/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ltransformer_decoder_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
:transformer_decoder_3/layer_normalization/moments/varianceMeanGtransformer_decoder_3/layer_normalization/moments/SquaredDifference:z:0Utransformer_decoder_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(~
9transformer_decoder_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
7transformer_decoder_3/layer_normalization/batchnorm/addAddV2Ctransformer_decoder_3/layer_normalization/moments/variance:output:0Btransformer_decoder_3/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/RsqrtRsqrt;transformer_decoder_3/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Ftransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_decoder_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_decoder_3/layer_normalization/batchnorm/mulMul=transformer_decoder_3/layer_normalization/batchnorm/Rsqrt:y:0Ntransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/mul_1Multransformer_decoder_3/add:z:0;transformer_decoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/mul_2Mul?transformer_decoder_3/layer_normalization/moments/mean:output:0;transformer_decoder_3/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Btransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKtransformer_decoder_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
7transformer_decoder_3/layer_normalization/batchnorm/subSubJtransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp:value:0=transformer_decoder_3/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
9transformer_decoder_3/layer_normalization/batchnorm/add_1AddV2=transformer_decoder_3/layer_normalization/batchnorm/mul_1:z:0;transformer_decoder_3/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ?
4transformer_decoder_3/dense/Tensordot/ReadVariableOpReadVariableOp=transformer_decoder_3_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0t
*transformer_decoder_3/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:{
*transformer_decoder_3/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
+transformer_decoder_3/dense/Tensordot/ShapeShape=transformer_decoder_3/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:u
3transformer_decoder_3/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_decoder_3/dense/Tensordot/GatherV2GatherV24transformer_decoder_3/dense/Tensordot/Shape:output:03transformer_decoder_3/dense/Tensordot/free:output:0<transformer_decoder_3/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
5transformer_decoder_3/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_decoder_3/dense/Tensordot/GatherV2_1GatherV24transformer_decoder_3/dense/Tensordot/Shape:output:03transformer_decoder_3/dense/Tensordot/axes:output:0>transformer_decoder_3/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
+transformer_decoder_3/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
*transformer_decoder_3/dense/Tensordot/ProdProd7transformer_decoder_3/dense/Tensordot/GatherV2:output:04transformer_decoder_3/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: w
-transformer_decoder_3/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_decoder_3/dense/Tensordot/Prod_1Prod9transformer_decoder_3/dense/Tensordot/GatherV2_1:output:06transformer_decoder_3/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: s
1transformer_decoder_3/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
,transformer_decoder_3/dense/Tensordot/concatConcatV23transformer_decoder_3/dense/Tensordot/free:output:03transformer_decoder_3/dense/Tensordot/axes:output:0:transformer_decoder_3/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
+transformer_decoder_3/dense/Tensordot/stackPack3transformer_decoder_3/dense/Tensordot/Prod:output:05transformer_decoder_3/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
/transformer_decoder_3/dense/Tensordot/transpose	Transpose=transformer_decoder_3/layer_normalization/batchnorm/add_1:z:05transformer_decoder_3/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
-transformer_decoder_3/dense/Tensordot/ReshapeReshape3transformer_decoder_3/dense/Tensordot/transpose:y:04transformer_decoder_3/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
,transformer_decoder_3/dense/Tensordot/MatMulMatMul6transformer_decoder_3/dense/Tensordot/Reshape:output:0<transformer_decoder_3/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
-transformer_decoder_3/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:u
3transformer_decoder_3/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_decoder_3/dense/Tensordot/concat_1ConcatV27transformer_decoder_3/dense/Tensordot/GatherV2:output:06transformer_decoder_3/dense/Tensordot/Const_2:output:0<transformer_decoder_3/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
%transformer_decoder_3/dense/TensordotReshape6transformer_decoder_3/dense/Tensordot/MatMul:product:07transformer_decoder_3/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
2transformer_decoder_3/dense/BiasAdd/ReadVariableOpReadVariableOp;transformer_decoder_3_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#transformer_decoder_3/dense/BiasAddBiasAdd.transformer_decoder_3/dense/Tensordot:output:0:transformer_decoder_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
 transformer_decoder_3/dense/ReluRelu,transformer_decoder_3/dense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
6transformer_decoder_3/dense_1/Tensordot/ReadVariableOpReadVariableOp?transformer_decoder_3_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0v
,transformer_decoder_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,transformer_decoder_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ?
-transformer_decoder_3/dense_1/Tensordot/ShapeShape.transformer_decoder_3/dense/Relu:activations:0*
T0*
_output_shapes
:w
5transformer_decoder_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_decoder_3/dense_1/Tensordot/GatherV2GatherV26transformer_decoder_3/dense_1/Tensordot/Shape:output:05transformer_decoder_3/dense_1/Tensordot/free:output:0>transformer_decoder_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7transformer_decoder_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2transformer_decoder_3/dense_1/Tensordot/GatherV2_1GatherV26transformer_decoder_3/dense_1/Tensordot/Shape:output:05transformer_decoder_3/dense_1/Tensordot/axes:output:0@transformer_decoder_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-transformer_decoder_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,transformer_decoder_3/dense_1/Tensordot/ProdProd9transformer_decoder_3/dense_1/Tensordot/GatherV2:output:06transformer_decoder_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/transformer_decoder_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
.transformer_decoder_3/dense_1/Tensordot/Prod_1Prod;transformer_decoder_3/dense_1/Tensordot/GatherV2_1:output:08transformer_decoder_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3transformer_decoder_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.transformer_decoder_3/dense_1/Tensordot/concatConcatV25transformer_decoder_3/dense_1/Tensordot/free:output:05transformer_decoder_3/dense_1/Tensordot/axes:output:0<transformer_decoder_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:?
-transformer_decoder_3/dense_1/Tensordot/stackPack5transformer_decoder_3/dense_1/Tensordot/Prod:output:07transformer_decoder_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:?
1transformer_decoder_3/dense_1/Tensordot/transpose	Transpose.transformer_decoder_3/dense/Relu:activations:07transformer_decoder_3/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? ?
/transformer_decoder_3/dense_1/Tensordot/ReshapeReshape5transformer_decoder_3/dense_1/Tensordot/transpose:y:06transformer_decoder_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:???????????????????
.transformer_decoder_3/dense_1/Tensordot/MatMulMatMul8transformer_decoder_3/dense_1/Tensordot/Reshape:output:0>transformer_decoder_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y
/transformer_decoder_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:w
5transformer_decoder_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0transformer_decoder_3/dense_1/Tensordot/concat_1ConcatV29transformer_decoder_3/dense_1/Tensordot/GatherV2:output:08transformer_decoder_3/dense_1/Tensordot/Const_2:output:0>transformer_decoder_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
'transformer_decoder_3/dense_1/TensordotReshape8transformer_decoder_3/dense_1/Tensordot/MatMul:product:09transformer_decoder_3/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? ?
4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp=transformer_decoder_3_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
%transformer_decoder_3/dense_1/BiasAddBiasAdd0transformer_decoder_3/dense_1/Tensordot:output:0<transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
(transformer_decoder_3/dropout_1/IdentityIdentity.transformer_decoder_3/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
transformer_decoder_3/add_1AddV2=transformer_decoder_3/layer_normalization/batchnorm/add_1:z:01transformer_decoder_3/dropout_1/Identity:output:0*
T0*+
_output_shapes
:????????? ?
Jtransformer_decoder_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
8transformer_decoder_3/layer_normalization_1/moments/meanMeantransformer_decoder_3/add_1:z:0Stransformer_decoder_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
@transformer_decoder_3/layer_normalization_1/moments/StopGradientStopGradientAtransformer_decoder_3/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:????????? ?
Etransformer_decoder_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_decoder_3/add_1:z:0Itransformer_decoder_3/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? ?
Ntransformer_decoder_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:?
<transformer_decoder_3/layer_normalization_1/moments/varianceMeanItransformer_decoder_3/layer_normalization_1/moments/SquaredDifference:z:0Wtransformer_decoder_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:????????? *
	keep_dims(?
;transformer_decoder_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
9transformer_decoder_3/layer_normalization_1/batchnorm/addAddV2Etransformer_decoder_3/layer_normalization_1/moments/variance:output:0Dtransformer_decoder_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/RsqrtRsqrt=transformer_decoder_3/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:????????? ?
Htransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_decoder_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_decoder_3/layer_normalization_1/batchnorm/mulMul?transformer_decoder_3/layer_normalization_1/batchnorm/Rsqrt:y:0Ptransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/mul_1Multransformer_decoder_3/add_1:z:0=transformer_decoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/mul_2MulAtransformer_decoder_3/layer_normalization_1/moments/mean:output:0=transformer_decoder_3/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ?
Dtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMtransformer_decoder_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0?
9transformer_decoder_3/layer_normalization_1/batchnorm/subSubLtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0?transformer_decoder_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:????????? ?
;transformer_decoder_3/layer_normalization_1/batchnorm/add_1AddV2?transformer_decoder_3/layer_normalization_1/batchnorm/mul_1:z:0=transformer_decoder_3/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? s
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_2/MeanMean?transformer_decoder_3/layer_normalization_1/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_4/MatMulMatMul(global_average_pooling1d_2/Mean:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? f
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? h
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^embedding_3/embedding_lookupE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2B^token_and_position_embedding_2/position_embedding3/ReadVariableOpA^token_and_position_embedding_2/token_embedding3/embedding_lookup3^transformer_decoder_3/dense/BiasAdd/ReadVariableOp5^transformer_decoder_3/dense/Tensordot/ReadVariableOp5^transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp7^transformer_decoder_3/dense_1/Tensordot/ReadVariableOpC^transformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpG^transformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpE^transformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpI^transformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpO^transformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpY^transformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^transformer_decoder_3/multi_head_attention/key/add/ReadVariableOpL^transformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^transformer_decoder_3/multi_head_attention/query/add/ReadVariableOpN^transformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^transformer_decoder_3/multi_head_attention/value/add/ReadVariableOpN^transformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp3^transformer_encoder_3/dense/BiasAdd/ReadVariableOp5^transformer_encoder_3/dense/Tensordot/ReadVariableOp5^transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp7^transformer_encoder_3/dense_1/Tensordot/ReadVariableOpC^transformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpG^transformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpE^transformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpI^transformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpO^transformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpY^transformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^transformer_encoder_3/multi_head_attention/key/add/ReadVariableOpL^transformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^transformer_encoder_3/multi_head_attention/query/add/ReadVariableOpN^transformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^transformer_encoder_3/multi_head_attention/value/add/ReadVariableOpN^transformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22?
Atoken_and_position_embedding_2/position_embedding3/ReadVariableOpAtoken_and_position_embedding_2/position_embedding3/ReadVariableOp2?
@token_and_position_embedding_2/token_embedding3/embedding_lookup@token_and_position_embedding_2/token_embedding3/embedding_lookup2h
2transformer_decoder_3/dense/BiasAdd/ReadVariableOp2transformer_decoder_3/dense/BiasAdd/ReadVariableOp2l
4transformer_decoder_3/dense/Tensordot/ReadVariableOp4transformer_decoder_3/dense/Tensordot/ReadVariableOp2l
4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp4transformer_decoder_3/dense_1/BiasAdd/ReadVariableOp2p
6transformer_decoder_3/dense_1/Tensordot/ReadVariableOp6transformer_decoder_3/dense_1/Tensordot/ReadVariableOp2?
Btransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOpBtransformer_decoder_3/layer_normalization/batchnorm/ReadVariableOp2?
Ftransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOpFtransformer_decoder_3/layer_normalization/batchnorm/mul/ReadVariableOp2?
Dtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOpDtransformer_decoder_3/layer_normalization_1/batchnorm/ReadVariableOp2?
Htransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpHtransformer_decoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Ntransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOpNtransformer_decoder_3/multi_head_attention/attention_output/add/ReadVariableOp2?
Xtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXtransformer_decoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_decoder_3/multi_head_attention/key/add/ReadVariableOpAtransformer_decoder_3/multi_head_attention/key/add/ReadVariableOp2?
Ktransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpKtransformer_decoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_decoder_3/multi_head_attention/query/add/ReadVariableOpCtransformer_decoder_3/multi_head_attention/query/add/ReadVariableOp2?
Mtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpMtransformer_decoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_decoder_3/multi_head_attention/value/add/ReadVariableOpCtransformer_decoder_3/multi_head_attention/value/add/ReadVariableOp2?
Mtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpMtransformer_decoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp2h
2transformer_encoder_3/dense/BiasAdd/ReadVariableOp2transformer_encoder_3/dense/BiasAdd/ReadVariableOp2l
4transformer_encoder_3/dense/Tensordot/ReadVariableOp4transformer_encoder_3/dense/Tensordot/ReadVariableOp2l
4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp4transformer_encoder_3/dense_1/BiasAdd/ReadVariableOp2p
6transformer_encoder_3/dense_1/Tensordot/ReadVariableOp6transformer_encoder_3/dense_1/Tensordot/ReadVariableOp2?
Btransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOpBtransformer_encoder_3/layer_normalization/batchnorm/ReadVariableOp2?
Ftransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOpFtransformer_encoder_3/layer_normalization/batchnorm/mul/ReadVariableOp2?
Dtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOpDtransformer_encoder_3/layer_normalization_1/batchnorm/ReadVariableOp2?
Htransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOpHtransformer_encoder_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2?
Ntransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOpNtransformer_encoder_3/multi_head_attention/attention_output/add/ReadVariableOp2?
Xtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXtransformer_encoder_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2?
Atransformer_encoder_3/multi_head_attention/key/add/ReadVariableOpAtransformer_encoder_3/multi_head_attention/key/add/ReadVariableOp2?
Ktransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpKtransformer_encoder_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2?
Ctransformer_encoder_3/multi_head_attention/query/add/ReadVariableOpCtransformer_encoder_3/multi_head_attention/query/add/ReadVariableOp2?
Mtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpMtransformer_encoder_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2?
Ctransformer_encoder_3/multi_head_attention/value/add/ReadVariableOpCtransformer_encoder_3/multi_head_attention/value/add/ReadVariableOp2?
Mtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpMtransformer_encoder_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:Q M
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
?
6__inference_transformer_decoder_3_layer_call_fn_336183
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
	unknown_9:

unknown_10:

unknown_11:

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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333245s
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
?
?
__inference_save_fn_336690
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

::P
add/yConst*
_output_shapes
: *
dtype0*
valueB B
table-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: T
add_1/yConst*
_output_shapes
: *
dtype0*
valueB Btable-valuesO
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
ͩ
?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333038

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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
?
?
__inference_<lambda>_3367069
5key_value_init318608_lookuptableimportv2_table_handle1
-key_value_init318608_lookuptableimportv2_keys3
/key_value_init318608_lookuptableimportv2_values	
identity??(key_value_init318608/LookupTableImportV2?
(key_value_init318608/LookupTableImportV2LookupTableImportV25key_value_init318608_lookuptableimportv2_table_handle-key_value_init318608_lookuptableimportv2_keys/key_value_init318608_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^key_value_init318608/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2T
(key_value_init318608/LookupTableImportV2(key_value_init318608/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?

(__inference_model_2_layer_call_fn_333404

phrase

token_role
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?
	unknown_4: 
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21: 

unknown_22:

unknown_23: 

unknown_24:

unknown_25: 

unknown_26:

unknown_27: 

unknown_28:

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:@

unknown_39:@

unknown_40:@ 

unknown_41: 
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
unknown_41*8
Tin1
/2-		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *I
_read_only_resource_inputs+
)'	
 !"#$%&'()*+,*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_333315o
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
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namePhrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
Token_role:
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
6__inference_transformer_encoder_3_layer_call_fn_335871

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
	unknown_9:

unknown_10:

unknown_11:

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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333881s
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
??
?
C__inference_model_2_layer_call_and_return_conditional_losses_334491

phrase

token_roleU
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	8
%token_and_position_embedding_2_334404:	?7
%token_and_position_embedding_2_334406: $
embedding_3_334409:2
transformer_encoder_3_334413:.
transformer_encoder_3_334415:2
transformer_encoder_3_334417:.
transformer_encoder_3_334419:2
transformer_encoder_3_334421:.
transformer_encoder_3_334423:2
transformer_encoder_3_334425:*
transformer_encoder_3_334427:*
transformer_encoder_3_334429:*
transformer_encoder_3_334431:.
transformer_encoder_3_334433:*
transformer_encoder_3_334435:.
transformer_encoder_3_334437:*
transformer_encoder_3_334439:*
transformer_encoder_3_334441:*
transformer_encoder_3_334443:2
transformer_decoder_3_334446:.
transformer_decoder_3_334448:2
transformer_decoder_3_334450:.
transformer_decoder_3_334452:2
transformer_decoder_3_334454:.
transformer_decoder_3_334456:2
transformer_decoder_3_334458:*
transformer_decoder_3_334460:*
transformer_decoder_3_334462:*
transformer_decoder_3_334464:.
transformer_decoder_3_334466:*
transformer_decoder_3_334468:.
transformer_decoder_3_334470:*
transformer_decoder_3_334472:*
transformer_decoder_3_334474:*
transformer_decoder_3_334476: 
dense_4_334480:@
dense_4_334482:@ 
dense_5_334485:@ 
dense_5_334487: 
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_3/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?6token_and_position_embedding_2/StatefulPartitionedCall?-transformer_decoder_3/StatefulPartitionedCall?-transformer_encoder_3/StatefulPartitionedCall{
text_vectorization/SqueezeSqueezephrase*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0%token_and_position_embedding_2_334404%token_and_position_embedding_2_334406*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_332883?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCall
token_roleembedding_3_334409*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_332899?
add_2/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_332909?
-transformer_encoder_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_encoder_3_334413transformer_encoder_3_334415transformer_encoder_3_334417transformer_encoder_3_334419transformer_encoder_3_334421transformer_encoder_3_334423transformer_encoder_3_334425transformer_encoder_3_334427transformer_encoder_3_334429transformer_encoder_3_334431transformer_encoder_3_334433transformer_encoder_3_334435transformer_encoder_3_334437transformer_encoder_3_334439transformer_encoder_3_334441transformer_encoder_3_334443*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333038?
-transformer_decoder_3/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_3/StatefulPartitionedCall:output:0transformer_decoder_3_334446transformer_decoder_3_334448transformer_decoder_3_334450transformer_decoder_3_334452transformer_decoder_3_334454transformer_decoder_3_334456transformer_decoder_3_334458transformer_decoder_3_334460transformer_decoder_3_334462transformer_decoder_3_334464transformer_decoder_3_334466transformer_decoder_3_334468transformer_decoder_3_334470transformer_decoder_3_334472transformer_decoder_3_334474transformer_decoder_3_334476*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333245?
*global_average_pooling1d_2/PartitionedCallPartitionedCall6transformer_decoder_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_332798?
dense_4/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_4_334480dense_4_334482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_333291?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_334485dense_5_334487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_333308w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV27^token_and_position_embedding_2/StatefulPartitionedCall.^transformer_decoder_3/StatefulPartitionedCall.^transformer_encoder_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2^
-transformer_decoder_3/StatefulPartitionedCall-transformer_decoder_3/StatefulPartitionedCall2^
-transformer_encoder_3/StatefulPartitionedCall-transformer_encoder_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_namePhrase:SO
'
_output_shapes
:????????? 
$
_user_specified_name
Token_role:
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
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333658
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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
??
?
C__inference_model_2_layer_call_and_return_conditional_losses_333315

inputs
inputs_1U
Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handleV
Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	8
%token_and_position_embedding_2_332884:	?7
%token_and_position_embedding_2_332886: $
embedding_3_332900:2
transformer_encoder_3_333039:.
transformer_encoder_3_333041:2
transformer_encoder_3_333043:.
transformer_encoder_3_333045:2
transformer_encoder_3_333047:.
transformer_encoder_3_333049:2
transformer_encoder_3_333051:*
transformer_encoder_3_333053:*
transformer_encoder_3_333055:*
transformer_encoder_3_333057:.
transformer_encoder_3_333059:*
transformer_encoder_3_333061:.
transformer_encoder_3_333063:*
transformer_encoder_3_333065:*
transformer_encoder_3_333067:*
transformer_encoder_3_333069:2
transformer_decoder_3_333246:.
transformer_decoder_3_333248:2
transformer_decoder_3_333250:.
transformer_decoder_3_333252:2
transformer_decoder_3_333254:.
transformer_decoder_3_333256:2
transformer_decoder_3_333258:*
transformer_decoder_3_333260:*
transformer_decoder_3_333262:*
transformer_decoder_3_333264:.
transformer_decoder_3_333266:*
transformer_decoder_3_333268:.
transformer_decoder_3_333270:*
transformer_decoder_3_333272:*
transformer_decoder_3_333274:*
transformer_decoder_3_333276: 
dense_4_333292:@
dense_4_333294:@ 
dense_5_333309:@ 
dense_5_333311: 
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?#embedding_3/StatefulPartitionedCall?Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2?6token_and_position_embedding_2/StatefulPartitionedCall?-transformer_decoder_3/StatefulPartitionedCall?-transformer_encoder_3/StatefulPartitionedCall{
text_vectorization/SqueezeSqueezeinputs*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Qtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Rtext_vectorization_string_lookup_hash_table_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tMtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????        ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:????????? *
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0%token_and_position_embedding_2_332884%token_and_position_embedding_2_332886*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_332883?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_3_332900*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_3_layer_call_and_return_conditional_losses_332899?
add_2/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_332909?
-transformer_encoder_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_encoder_3_333039transformer_encoder_3_333041transformer_encoder_3_333043transformer_encoder_3_333045transformer_encoder_3_333047transformer_encoder_3_333049transformer_encoder_3_333051transformer_encoder_3_333053transformer_encoder_3_333055transformer_encoder_3_333057transformer_encoder_3_333059transformer_encoder_3_333061transformer_encoder_3_333063transformer_encoder_3_333065transformer_encoder_3_333067transformer_encoder_3_333069*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333038?
-transformer_decoder_3/StatefulPartitionedCallStatefulPartitionedCall6transformer_encoder_3/StatefulPartitionedCall:output:0transformer_decoder_3_333246transformer_decoder_3_333248transformer_decoder_3_333250transformer_decoder_3_333252transformer_decoder_3_333254transformer_decoder_3_333256transformer_decoder_3_333258transformer_decoder_3_333260transformer_decoder_3_333262transformer_decoder_3_333264transformer_decoder_3_333266transformer_decoder_3_333268transformer_decoder_3_333270transformer_decoder_3_333272transformer_decoder_3_333274transformer_decoder_3_333276*
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
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_333245?
*global_average_pooling1d_2/PartitionedCallPartitionedCall6transformer_decoder_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_332798?
dense_4/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0dense_4_333292dense_4_333294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_333291?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_333309dense_5_333311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_333308w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallE^text_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV27^token_and_position_embedding_2/StatefulPartitionedCall.^transformer_decoder_3/StatefulPartitionedCall.^transformer_encoder_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes~
|:?????????:????????? : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2?
Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV2Dtext_vectorization/string_lookup/hash_table_Lookup/LookupTableFindV22p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2^
-transformer_decoder_3/StatefulPartitionedCall-transformer_decoder_3/StatefulPartitionedCall2^
-transformer_encoder_3/StatefulPartitionedCall-transformer_encoder_3/StatefulPartitionedCall:O K
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
??
?P
__inference__traced_save_337127
file_prefix5
1savev2_embedding_3_embeddings_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableopY
Usavev2_token_and_position_embedding_2_token_embedding3_embeddings_read_readvariableop\
Xsavev2_token_and_position_embedding_2_position_embedding3_embeddings_read_readvariableopV
Rsavev2_transformer_encoder_3_multi_head_attention_query_kernel_read_readvariableopT
Psavev2_transformer_encoder_3_multi_head_attention_query_bias_read_readvariableopT
Psavev2_transformer_encoder_3_multi_head_attention_key_kernel_read_readvariableopR
Nsavev2_transformer_encoder_3_multi_head_attention_key_bias_read_readvariableopV
Rsavev2_transformer_encoder_3_multi_head_attention_value_kernel_read_readvariableopT
Psavev2_transformer_encoder_3_multi_head_attention_value_bias_read_readvariableopa
]savev2_transformer_encoder_3_multi_head_attention_attention_output_kernel_read_readvariableop_
[savev2_transformer_encoder_3_multi_head_attention_attention_output_bias_read_readvariableopN
Jsavev2_transformer_encoder_3_layer_normalization_gamma_read_readvariableopM
Isavev2_transformer_encoder_3_layer_normalization_beta_read_readvariableopP
Lsavev2_transformer_encoder_3_layer_normalization_1_gamma_read_readvariableopO
Ksavev2_transformer_encoder_3_layer_normalization_1_beta_read_readvariableopA
=savev2_transformer_encoder_3_dense_kernel_read_readvariableop?
;savev2_transformer_encoder_3_dense_bias_read_readvariableopC
?savev2_transformer_encoder_3_dense_1_kernel_read_readvariableopA
=savev2_transformer_encoder_3_dense_1_bias_read_readvariableopV
Rsavev2_transformer_decoder_3_multi_head_attention_query_kernel_read_readvariableopT
Psavev2_transformer_decoder_3_multi_head_attention_query_bias_read_readvariableopT
Psavev2_transformer_decoder_3_multi_head_attention_key_kernel_read_readvariableopR
Nsavev2_transformer_decoder_3_multi_head_attention_key_bias_read_readvariableopV
Rsavev2_transformer_decoder_3_multi_head_attention_value_kernel_read_readvariableopT
Psavev2_transformer_decoder_3_multi_head_attention_value_bias_read_readvariableopa
]savev2_transformer_decoder_3_multi_head_attention_attention_output_kernel_read_readvariableop_
[savev2_transformer_decoder_3_multi_head_attention_attention_output_bias_read_readvariableopN
Jsavev2_transformer_decoder_3_layer_normalization_gamma_read_readvariableopM
Isavev2_transformer_decoder_3_layer_normalization_beta_read_readvariableopP
Lsavev2_transformer_decoder_3_layer_normalization_1_gamma_read_readvariableopO
Ksavev2_transformer_decoder_3_layer_normalization_1_beta_read_readvariableopA
=savev2_transformer_decoder_3_dense_kernel_read_readvariableop?
;savev2_transformer_decoder_3_dense_bias_read_readvariableopC
?savev2_transformer_decoder_3_dense_1_kernel_read_readvariableopA
=savev2_transformer_decoder_3_dense_1_bias_read_readvariableop(
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
 savev2_count_read_readvariableop<
8savev2_adam_embedding_3_embeddings_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop`
\savev2_adam_token_and_position_embedding_2_token_embedding3_embeddings_m_read_readvariableopc
_savev2_adam_token_and_position_embedding_2_position_embedding3_embeddings_m_read_readvariableop]
Ysavev2_adam_transformer_encoder_3_multi_head_attention_query_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_encoder_3_multi_head_attention_query_bias_m_read_readvariableop[
Wsavev2_adam_transformer_encoder_3_multi_head_attention_key_kernel_m_read_readvariableopY
Usavev2_adam_transformer_encoder_3_multi_head_attention_key_bias_m_read_readvariableop]
Ysavev2_adam_transformer_encoder_3_multi_head_attention_value_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_encoder_3_multi_head_attention_value_bias_m_read_readvariableoph
dsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_m_read_readvariableopf
bsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_m_read_readvariableopU
Qsavev2_adam_transformer_encoder_3_layer_normalization_gamma_m_read_readvariableopT
Psavev2_adam_transformer_encoder_3_layer_normalization_beta_m_read_readvariableopW
Ssavev2_adam_transformer_encoder_3_layer_normalization_1_gamma_m_read_readvariableopV
Rsavev2_adam_transformer_encoder_3_layer_normalization_1_beta_m_read_readvariableopH
Dsavev2_adam_transformer_encoder_3_dense_kernel_m_read_readvariableopF
Bsavev2_adam_transformer_encoder_3_dense_bias_m_read_readvariableopJ
Fsavev2_adam_transformer_encoder_3_dense_1_kernel_m_read_readvariableopH
Dsavev2_adam_transformer_encoder_3_dense_1_bias_m_read_readvariableop]
Ysavev2_adam_transformer_decoder_3_multi_head_attention_query_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_decoder_3_multi_head_attention_query_bias_m_read_readvariableop[
Wsavev2_adam_transformer_decoder_3_multi_head_attention_key_kernel_m_read_readvariableopY
Usavev2_adam_transformer_decoder_3_multi_head_attention_key_bias_m_read_readvariableop]
Ysavev2_adam_transformer_decoder_3_multi_head_attention_value_kernel_m_read_readvariableop[
Wsavev2_adam_transformer_decoder_3_multi_head_attention_value_bias_m_read_readvariableoph
dsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_m_read_readvariableopf
bsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_m_read_readvariableopU
Qsavev2_adam_transformer_decoder_3_layer_normalization_gamma_m_read_readvariableopT
Psavev2_adam_transformer_decoder_3_layer_normalization_beta_m_read_readvariableopW
Ssavev2_adam_transformer_decoder_3_layer_normalization_1_gamma_m_read_readvariableopV
Rsavev2_adam_transformer_decoder_3_layer_normalization_1_beta_m_read_readvariableopH
Dsavev2_adam_transformer_decoder_3_dense_kernel_m_read_readvariableopF
Bsavev2_adam_transformer_decoder_3_dense_bias_m_read_readvariableopJ
Fsavev2_adam_transformer_decoder_3_dense_1_kernel_m_read_readvariableopH
Dsavev2_adam_transformer_decoder_3_dense_1_bias_m_read_readvariableop<
8savev2_adam_embedding_3_embeddings_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop`
\savev2_adam_token_and_position_embedding_2_token_embedding3_embeddings_v_read_readvariableopc
_savev2_adam_token_and_position_embedding_2_position_embedding3_embeddings_v_read_readvariableop]
Ysavev2_adam_transformer_encoder_3_multi_head_attention_query_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_encoder_3_multi_head_attention_query_bias_v_read_readvariableop[
Wsavev2_adam_transformer_encoder_3_multi_head_attention_key_kernel_v_read_readvariableopY
Usavev2_adam_transformer_encoder_3_multi_head_attention_key_bias_v_read_readvariableop]
Ysavev2_adam_transformer_encoder_3_multi_head_attention_value_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_encoder_3_multi_head_attention_value_bias_v_read_readvariableoph
dsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_v_read_readvariableopf
bsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_v_read_readvariableopU
Qsavev2_adam_transformer_encoder_3_layer_normalization_gamma_v_read_readvariableopT
Psavev2_adam_transformer_encoder_3_layer_normalization_beta_v_read_readvariableopW
Ssavev2_adam_transformer_encoder_3_layer_normalization_1_gamma_v_read_readvariableopV
Rsavev2_adam_transformer_encoder_3_layer_normalization_1_beta_v_read_readvariableopH
Dsavev2_adam_transformer_encoder_3_dense_kernel_v_read_readvariableopF
Bsavev2_adam_transformer_encoder_3_dense_bias_v_read_readvariableopJ
Fsavev2_adam_transformer_encoder_3_dense_1_kernel_v_read_readvariableopH
Dsavev2_adam_transformer_encoder_3_dense_1_bias_v_read_readvariableop]
Ysavev2_adam_transformer_decoder_3_multi_head_attention_query_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_decoder_3_multi_head_attention_query_bias_v_read_readvariableop[
Wsavev2_adam_transformer_decoder_3_multi_head_attention_key_kernel_v_read_readvariableopY
Usavev2_adam_transformer_decoder_3_multi_head_attention_key_bias_v_read_readvariableop]
Ysavev2_adam_transformer_decoder_3_multi_head_attention_value_kernel_v_read_readvariableop[
Wsavev2_adam_transformer_decoder_3_multi_head_attention_value_bias_v_read_readvariableoph
dsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_v_read_readvariableopf
bsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_v_read_readvariableopU
Qsavev2_adam_transformer_decoder_3_layer_normalization_gamma_v_read_readvariableopT
Psavev2_adam_transformer_decoder_3_layer_normalization_beta_v_read_readvariableopW
Ssavev2_adam_transformer_decoder_3_layer_normalization_1_gamma_v_read_readvariableopV
Rsavev2_adam_transformer_decoder_3_layer_normalization_1_beta_v_read_readvariableopH
Dsavev2_adam_transformer_decoder_3_dense_kernel_v_read_readvariableopF
Bsavev2_adam_transformer_decoder_3_dense_bias_v_read_readvariableopJ
Fsavev2_adam_transformer_decoder_3_dense_1_kernel_v_read_readvariableopH
Dsavev2_adam_transformer_decoder_3_dense_1_bias_v_read_readvariableop
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
: ?=
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?=
value?=B?=?B:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?M
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_3_embeddings_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopUsavev2_token_and_position_embedding_2_token_embedding3_embeddings_read_readvariableopXsavev2_token_and_position_embedding_2_position_embedding3_embeddings_read_readvariableopRsavev2_transformer_encoder_3_multi_head_attention_query_kernel_read_readvariableopPsavev2_transformer_encoder_3_multi_head_attention_query_bias_read_readvariableopPsavev2_transformer_encoder_3_multi_head_attention_key_kernel_read_readvariableopNsavev2_transformer_encoder_3_multi_head_attention_key_bias_read_readvariableopRsavev2_transformer_encoder_3_multi_head_attention_value_kernel_read_readvariableopPsavev2_transformer_encoder_3_multi_head_attention_value_bias_read_readvariableop]savev2_transformer_encoder_3_multi_head_attention_attention_output_kernel_read_readvariableop[savev2_transformer_encoder_3_multi_head_attention_attention_output_bias_read_readvariableopJsavev2_transformer_encoder_3_layer_normalization_gamma_read_readvariableopIsavev2_transformer_encoder_3_layer_normalization_beta_read_readvariableopLsavev2_transformer_encoder_3_layer_normalization_1_gamma_read_readvariableopKsavev2_transformer_encoder_3_layer_normalization_1_beta_read_readvariableop=savev2_transformer_encoder_3_dense_kernel_read_readvariableop;savev2_transformer_encoder_3_dense_bias_read_readvariableop?savev2_transformer_encoder_3_dense_1_kernel_read_readvariableop=savev2_transformer_encoder_3_dense_1_bias_read_readvariableopRsavev2_transformer_decoder_3_multi_head_attention_query_kernel_read_readvariableopPsavev2_transformer_decoder_3_multi_head_attention_query_bias_read_readvariableopPsavev2_transformer_decoder_3_multi_head_attention_key_kernel_read_readvariableopNsavev2_transformer_decoder_3_multi_head_attention_key_bias_read_readvariableopRsavev2_transformer_decoder_3_multi_head_attention_value_kernel_read_readvariableopPsavev2_transformer_decoder_3_multi_head_attention_value_bias_read_readvariableop]savev2_transformer_decoder_3_multi_head_attention_attention_output_kernel_read_readvariableop[savev2_transformer_decoder_3_multi_head_attention_attention_output_bias_read_readvariableopJsavev2_transformer_decoder_3_layer_normalization_gamma_read_readvariableopIsavev2_transformer_decoder_3_layer_normalization_beta_read_readvariableopLsavev2_transformer_decoder_3_layer_normalization_1_gamma_read_readvariableopKsavev2_transformer_decoder_3_layer_normalization_1_beta_read_readvariableop=savev2_transformer_decoder_3_dense_kernel_read_readvariableop;savev2_transformer_decoder_3_dense_bias_read_readvariableop?savev2_transformer_decoder_3_dense_1_kernel_read_readvariableop=savev2_transformer_decoder_3_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop8savev2_adam_embedding_3_embeddings_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop\savev2_adam_token_and_position_embedding_2_token_embedding3_embeddings_m_read_readvariableop_savev2_adam_token_and_position_embedding_2_position_embedding3_embeddings_m_read_readvariableopYsavev2_adam_transformer_encoder_3_multi_head_attention_query_kernel_m_read_readvariableopWsavev2_adam_transformer_encoder_3_multi_head_attention_query_bias_m_read_readvariableopWsavev2_adam_transformer_encoder_3_multi_head_attention_key_kernel_m_read_readvariableopUsavev2_adam_transformer_encoder_3_multi_head_attention_key_bias_m_read_readvariableopYsavev2_adam_transformer_encoder_3_multi_head_attention_value_kernel_m_read_readvariableopWsavev2_adam_transformer_encoder_3_multi_head_attention_value_bias_m_read_readvariableopdsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_m_read_readvariableopbsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_m_read_readvariableopQsavev2_adam_transformer_encoder_3_layer_normalization_gamma_m_read_readvariableopPsavev2_adam_transformer_encoder_3_layer_normalization_beta_m_read_readvariableopSsavev2_adam_transformer_encoder_3_layer_normalization_1_gamma_m_read_readvariableopRsavev2_adam_transformer_encoder_3_layer_normalization_1_beta_m_read_readvariableopDsavev2_adam_transformer_encoder_3_dense_kernel_m_read_readvariableopBsavev2_adam_transformer_encoder_3_dense_bias_m_read_readvariableopFsavev2_adam_transformer_encoder_3_dense_1_kernel_m_read_readvariableopDsavev2_adam_transformer_encoder_3_dense_1_bias_m_read_readvariableopYsavev2_adam_transformer_decoder_3_multi_head_attention_query_kernel_m_read_readvariableopWsavev2_adam_transformer_decoder_3_multi_head_attention_query_bias_m_read_readvariableopWsavev2_adam_transformer_decoder_3_multi_head_attention_key_kernel_m_read_readvariableopUsavev2_adam_transformer_decoder_3_multi_head_attention_key_bias_m_read_readvariableopYsavev2_adam_transformer_decoder_3_multi_head_attention_value_kernel_m_read_readvariableopWsavev2_adam_transformer_decoder_3_multi_head_attention_value_bias_m_read_readvariableopdsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_m_read_readvariableopbsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_m_read_readvariableopQsavev2_adam_transformer_decoder_3_layer_normalization_gamma_m_read_readvariableopPsavev2_adam_transformer_decoder_3_layer_normalization_beta_m_read_readvariableopSsavev2_adam_transformer_decoder_3_layer_normalization_1_gamma_m_read_readvariableopRsavev2_adam_transformer_decoder_3_layer_normalization_1_beta_m_read_readvariableopDsavev2_adam_transformer_decoder_3_dense_kernel_m_read_readvariableopBsavev2_adam_transformer_decoder_3_dense_bias_m_read_readvariableopFsavev2_adam_transformer_decoder_3_dense_1_kernel_m_read_readvariableopDsavev2_adam_transformer_decoder_3_dense_1_bias_m_read_readvariableop8savev2_adam_embedding_3_embeddings_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop\savev2_adam_token_and_position_embedding_2_token_embedding3_embeddings_v_read_readvariableop_savev2_adam_token_and_position_embedding_2_position_embedding3_embeddings_v_read_readvariableopYsavev2_adam_transformer_encoder_3_multi_head_attention_query_kernel_v_read_readvariableopWsavev2_adam_transformer_encoder_3_multi_head_attention_query_bias_v_read_readvariableopWsavev2_adam_transformer_encoder_3_multi_head_attention_key_kernel_v_read_readvariableopUsavev2_adam_transformer_encoder_3_multi_head_attention_key_bias_v_read_readvariableopYsavev2_adam_transformer_encoder_3_multi_head_attention_value_kernel_v_read_readvariableopWsavev2_adam_transformer_encoder_3_multi_head_attention_value_bias_v_read_readvariableopdsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_kernel_v_read_readvariableopbsavev2_adam_transformer_encoder_3_multi_head_attention_attention_output_bias_v_read_readvariableopQsavev2_adam_transformer_encoder_3_layer_normalization_gamma_v_read_readvariableopPsavev2_adam_transformer_encoder_3_layer_normalization_beta_v_read_readvariableopSsavev2_adam_transformer_encoder_3_layer_normalization_1_gamma_v_read_readvariableopRsavev2_adam_transformer_encoder_3_layer_normalization_1_beta_v_read_readvariableopDsavev2_adam_transformer_encoder_3_dense_kernel_v_read_readvariableopBsavev2_adam_transformer_encoder_3_dense_bias_v_read_readvariableopFsavev2_adam_transformer_encoder_3_dense_1_kernel_v_read_readvariableopDsavev2_adam_transformer_encoder_3_dense_1_bias_v_read_readvariableopYsavev2_adam_transformer_decoder_3_multi_head_attention_query_kernel_v_read_readvariableopWsavev2_adam_transformer_decoder_3_multi_head_attention_query_bias_v_read_readvariableopWsavev2_adam_transformer_decoder_3_multi_head_attention_key_kernel_v_read_readvariableopUsavev2_adam_transformer_decoder_3_multi_head_attention_key_bias_v_read_readvariableopYsavev2_adam_transformer_decoder_3_multi_head_attention_value_kernel_v_read_readvariableopWsavev2_adam_transformer_decoder_3_multi_head_attention_value_bias_v_read_readvariableopdsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_kernel_v_read_readvariableopbsavev2_adam_transformer_decoder_3_multi_head_attention_attention_output_bias_v_read_readvariableopQsavev2_adam_transformer_decoder_3_layer_normalization_gamma_v_read_readvariableopPsavev2_adam_transformer_decoder_3_layer_normalization_beta_v_read_readvariableopSsavev2_adam_transformer_decoder_3_layer_normalization_1_gamma_v_read_readvariableopRsavev2_adam_transformer_decoder_3_layer_normalization_1_beta_v_read_readvariableopDsavev2_adam_transformer_decoder_3_dense_kernel_v_read_readvariableopBsavev2_adam_transformer_decoder_3_dense_bias_v_read_readvariableopFsavev2_adam_transformer_decoder_3_dense_1_kernel_v_read_readvariableopDsavev2_adam_transformer_decoder_3_dense_1_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?		?
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
?: ::@:@:@ : :	?: ::::::::::::::::::::::::::::::::: : : : : ::: : : : ::@:@:@ : :	?: ::::::::::::::::::::::::::::::::::@:@:@ : :	?: ::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :%!

_output_shapes
:	?:$ 

_output_shapes

: :($
"
_output_shapes
::$	 

_output_shapes

::(
$
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
:: 
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
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
:: 

_output_shapes
::  
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
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::(
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
: :,

_output_shapes
: :-

_output_shapes
::.

_output_shapes
::/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :$3 

_output_shapes

::$4 

_output_shapes

:@: 5

_output_shapes
:@:$6 

_output_shapes

:@ : 7

_output_shapes
: :%8!

_output_shapes
:	?:$9 

_output_shapes

: :(:$
"
_output_shapes
::$; 

_output_shapes

::(<$
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
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::(J$
"
_output_shapes
::$K 

_output_shapes

::(L$
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
:: Q

_output_shapes
:: R

_output_shapes
:: S

_output_shapes
:: T

_output_shapes
:: U

_output_shapes
::$V 

_output_shapes

:: W

_output_shapes
::$X 

_output_shapes

:: Y

_output_shapes
::$Z 

_output_shapes

::$[ 

_output_shapes

:@: \

_output_shapes
:@:$] 

_output_shapes

:@ : ^

_output_shapes
: :%_!

_output_shapes
:	?:$` 

_output_shapes

: :(a$
"
_output_shapes
::$b 

_output_shapes

::(c$
"
_output_shapes
::$d 

_output_shapes

::(e$
"
_output_shapes
::$f 

_output_shapes

::(g$
"
_output_shapes
:: h

_output_shapes
:: i

_output_shapes
:: j

_output_shapes
:: k

_output_shapes
:: l

_output_shapes
::$m 

_output_shapes

:: n

_output_shapes
::$o 

_output_shapes

:: p

_output_shapes
::(q$
"
_output_shapes
::$r 

_output_shapes

::(s$
"
_output_shapes
::$t 

_output_shapes

::(u$
"
_output_shapes
::$v 

_output_shapes

::(w$
"
_output_shapes
:: x

_output_shapes
:: y

_output_shapes
:: z

_output_shapes
:: {

_output_shapes
:: |

_output_shapes
::$} 

_output_shapes

:: ~

_output_shapes
::$ 

_output_shapes

::!?

_output_shapes
::?

_output_shapes
: 
?
?
?__inference_token_and_position_embedding_2_layer_call_fn_335741

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
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_332883s
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
?
;
__inference__creator_336643
identity??
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name318609*
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
?
m
A__inference_add_2_layer_call_and_return_conditional_losses_335797
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
ͩ
?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_335998

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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
?

?
C__inference_dense_5_layer_call_and_return_conditional_losses_336638

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
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
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
G__inference_embedding_3_layer_call_and_return_conditional_losses_332899

inputs)
embedding_lookup_332893:
identity??embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:????????? ?
embedding_lookupResourceGatherembedding_lookup_332893Cast:y:0*
Tindices0**
_class 
loc:@embedding_lookup/332893*+
_output_shapes
:????????? *
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/332893*+
_output_shapes
:????????? ?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:????????? w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:????????? Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:????????? : 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
/
__inference__initializer_336666
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
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_333881

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
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
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

:*
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
:?????????a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
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
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? `

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:????????? ?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
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
:????????? ?
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
Phrase/
serving_default_Phrase:0?????????
A

Token_role3
serving_default_Token_role:0????????? =
dense_52
StatefulPartitionedCall_1:0????????? tensorflow/serving/predict:??
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
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
P
	keras_api
_lookup_layer
_adapt_function"
_tf_keras_layer
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
token_embedding
position_embedding"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses
&
embeddings"
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
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

[kernel
\bias"
_tf_keras_layer
?
]1
^2
&3
_4
`5
a6
b7
c8
d9
e10
f11
g12
h13
i14
j15
k16
l17
m18
n19
o20
p21
q22
r23
s24
t25
u26
v27
w28
x29
y30
z31
{32
|33
}34
~35
S36
T37
[38
\39"
trackable_list_wrapper
?
]0
^1
&2
_3
`4
a5
b6
c7
d8
e9
f10
g11
h12
i13
j14
k15
l16
m17
n18
o19
p20
q21
r22
s23
t24
u25
v26
w27
x28
y29
z30
{31
|32
}33
~34
S35
T36
[37
\38"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
(__inference_model_2_layer_call_fn_333404
(__inference_model_2_layer_call_fn_334820
(__inference_model_2_layer_call_fn_334912
(__inference_model_2_layer_call_fn_334354?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
C__inference_model_2_layer_call_and_return_conditional_losses_335301
C__inference_model_2_layer_call_and_return_conditional_losses_335732
C__inference_model_2_layer_call_and_return_conditional_losses_334491
C__inference_model_2_layer_call_and_return_conditional_losses_334628?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
!__inference__wrapped_model_332788Phrase
Token_role"?
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
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate&m?Sm?Tm?[m?\m?]m?^m?_m?`m?am?bm?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?&v?Sv?Tv?[v?\v?]v?^v?_v?`v?av?bv?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?"
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
__inference_adapt_step_202704?
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
?__inference_token_and_position_embedding_2_layer_call_fn_335741?
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_335768?
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
]
embeddings"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
^
embeddings
^position_embeddings"
_tf_keras_layer
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_embedding_3_layer_call_fn_335775?
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
G__inference_embedding_3_layer_call_and_return_conditional_losses_335785?
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
(:&2embedding_3/embeddings
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
&__inference_add_2_layer_call_fn_335791?
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
A__inference_add_2_layer_call_and_return_conditional_losses_335797?
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
_0
`1
a2
b3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15"
trackable_list_wrapper
?
_0
`1
a2
b3
c4
d5
e6
f7
g8
h9
i10
j11
k12
l13
m14
n15"
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
6__inference_transformer_encoder_3_layer_call_fn_335834
6__inference_transformer_encoder_3_layer_call_fn_335871?
???
FullArgSpecK
argsC?@
jself
jinputs
jpadding_mask
jattention_mask

jtraining
varargs
 
varkw
 
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_335998
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_336146?
???
FullArgSpecK
argsC?@
jself
jinputs
jpadding_mask
jattention_mask

jtraining
varargs
 
varkw
 
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults
 
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
	ggamma
hbeta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	igamma
jbeta"
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

kkernel
lbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

mkernel
nbias"
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
o0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15"
trackable_list_wrapper
?
o0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_transformer_decoder_3_layer_call_fn_336183
6__inference_transformer_decoder_3_layer_call_fn_336220?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask

jtraining
varargs
 
varkw
 '
defaults?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336393
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336587?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask

jtraining
varargs
 
varkw
 '
defaults?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?	variables
?trainable_variables
?regularization_losses
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
	wgamma
xbeta"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis
	ygamma
zbeta"
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

{kernel
|bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

}kernel
~bias"
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
;__inference_global_average_pooling1d_2_layer_call_fn_336592?
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
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_336598?
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
(__inference_dense_4_layer_call_fn_336607?
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
C__inference_dense_4_layer_call_and_return_conditional_losses_336618?
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
 :@2dense_4/kernel
:@2dense_4/bias
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
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_5_layer_call_fn_336627?
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
C__inference_dense_5_layer_call_and_return_conditional_losses_336638?
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
 :@ 2dense_5/kernel
: 2dense_5/bias
M:K	?2:token_and_position_embedding_2/token_embedding3/embeddings
O:M 2=token_and_position_embedding_2/position_embedding3/embeddings
M:K27transformer_encoder_3/multi_head_attention/query/kernel
G:E25transformer_encoder_3/multi_head_attention/query/bias
K:I25transformer_encoder_3/multi_head_attention/key/kernel
E:C23transformer_encoder_3/multi_head_attention/key/bias
M:K27transformer_encoder_3/multi_head_attention/value/kernel
G:E25transformer_encoder_3/multi_head_attention/value/bias
X:V2Btransformer_encoder_3/multi_head_attention/attention_output/kernel
N:L2@transformer_encoder_3/multi_head_attention/attention_output/bias
=:;2/transformer_encoder_3/layer_normalization/gamma
<::2.transformer_encoder_3/layer_normalization/beta
?:=21transformer_encoder_3/layer_normalization_1/gamma
>:<20transformer_encoder_3/layer_normalization_1/beta
4:22"transformer_encoder_3/dense/kernel
.:,2 transformer_encoder_3/dense/bias
6:42$transformer_encoder_3/dense_1/kernel
0:.2"transformer_encoder_3/dense_1/bias
M:K27transformer_decoder_3/multi_head_attention/query/kernel
G:E25transformer_decoder_3/multi_head_attention/query/bias
K:I25transformer_decoder_3/multi_head_attention/key/kernel
E:C23transformer_decoder_3/multi_head_attention/key/bias
M:K27transformer_decoder_3/multi_head_attention/value/kernel
G:E25transformer_decoder_3/multi_head_attention/value/bias
X:V2Btransformer_decoder_3/multi_head_attention/attention_output/kernel
N:L2@transformer_decoder_3/multi_head_attention/attention_output/bias
=:;2/transformer_decoder_3/layer_normalization/gamma
<::2.transformer_decoder_3/layer_normalization/beta
?:=21transformer_decoder_3/layer_normalization_1/gamma
>:<20transformer_decoder_3/layer_normalization_1/beta
4:22"transformer_decoder_3/dense/kernel
.:,2 transformer_decoder_3/dense/bias
6:42$transformer_decoder_3/dense_1/kernel
0:.2"transformer_decoder_3/dense_1/bias
 "
trackable_list_wrapper
n
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
10"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_model_2_layer_call_fn_333404Phrase
Token_role"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_model_2_layer_call_fn_334820inputs/0inputs/1"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_model_2_layer_call_fn_334912inputs/0inputs/1"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_model_2_layer_call_fn_334354Phrase
Token_role"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_2_layer_call_and_return_conditional_losses_335301inputs/0inputs/1"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_2_layer_call_and_return_conditional_losses_335732inputs/0inputs/1"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_2_layer_call_and_return_conditional_losses_334491Phrase
Token_role"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_2_layer_call_and_return_conditional_losses_334628Phrase
Token_role"?
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
$__inference_signature_wrapper_334728Phrase
Token_role"?
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
 
"
_generic_user_object
j
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR jtf.StaticHashTable
O
?_create_resource
?_initialize
?_destroy_resourceR Z

 ??
?B?
__inference_adapt_step_202704iterator"?
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
 
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
?__inference_token_and_position_embedding_2_layer_call_fn_335741inputs"?
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
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_335768inputs"?
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
]0"
trackable_list_wrapper
'
]0"
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
^0"
trackable_list_wrapper
'
^0"
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
,__inference_embedding_3_layer_call_fn_335775inputs"?
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
G__inference_embedding_3_layer_call_and_return_conditional_losses_335785inputs"?
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
&__inference_add_2_layer_call_fn_335791inputs/0inputs/1"?
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
A__inference_add_2_layer_call_and_return_conditional_losses_335797inputs/0inputs/1"?
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
6__inference_transformer_encoder_3_layer_call_fn_335834inputs"?
???
FullArgSpecK
argsC?@
jself
jinputs
jpadding_mask
jattention_mask

jtraining
varargs
 
varkw
 
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
6__inference_transformer_encoder_3_layer_call_fn_335871inputs"?
???
FullArgSpecK
argsC?@
jself
jinputs
jpadding_mask
jattention_mask

jtraining
varargs
 
varkw
 
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_335998inputs"?
???
FullArgSpecK
argsC?@
jself
jinputs
jpadding_mask
jattention_mask

jtraining
varargs
 
varkw
 
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_336146inputs"?
???
FullArgSpecK
argsC?@
jself
jinputs
jpadding_mask
jattention_mask

jtraining
varargs
 
varkw
 
defaults?

 

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
X
_0
`1
a2
b3
c4
d5
e6
f7"
trackable_list_wrapper
X
_0
`1
a2
b3
c4
d5
e6
f7"
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
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
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

_kernel
`bias"
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

akernel
bbias"
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

ckernel
dbias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
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

ekernel
fbias"
_tf_keras_layer
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
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
6__inference_transformer_decoder_3_layer_call_fn_336183decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask

jtraining
varargs
 
varkw
 '
defaults?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
6__inference_transformer_decoder_3_layer_call_fn_336220decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask

jtraining
varargs
 
varkw
 '
defaults?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336393decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask

jtraining
varargs
 
varkw
 '
defaults?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336587decoder_sequence"?
???
FullArgSpec?
args???
jself
jdecoder_sequence
jencoder_sequence
jdecoder_padding_mask
jdecoder_attention_mask
jencoder_padding_mask
jencoder_attention_mask

jtraining
varargs
 
varkw
 '
defaults?
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
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
X
o0
p1
q2
r3
s4
t5
u6
v7"
trackable_list_wrapper
X
o0
p1
q2
r3
s4
t5
u6
v7"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpece
args]?Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults?

 

 
p 
p 

kwonlyargs? 
kwonlydefaults? 
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

okernel
pbias"
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

qkernel
rbias"
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

skernel
tbias"
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

ukernel
vbias"
_tf_keras_layer
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
y0
z1"
trackable_list_wrapper
.
y0
z1"
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
"
_generic_user_object
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
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
}0
~1"
trackable_list_wrapper
.
}0
~1"
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
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
;__inference_global_average_pooling1d_2_layer_call_fn_336592inputs"?
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
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_336598inputs"?
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
(__inference_dense_4_layer_call_fn_336607inputs"?
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
C__inference_dense_4_layer_call_and_return_conditional_losses_336618inputs"?
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
(__inference_dense_5_layer_call_fn_336627inputs"?
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
C__inference_dense_5_layer_call_and_return_conditional_losses_336638inputs"?
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
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
"
_generic_user_object
?
?trace_02?
__inference__creator_336643?
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
__inference__initializer_336651?
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
__inference__destroyer_336656?
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
__inference__creator_336661?
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
__inference__initializer_336666?
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
__inference__destroyer_336671?
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
?	variables
?trainable_variables
?regularization_losses
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
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

kwonlyargs? 
kwonlydefaults? 
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
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?B?
__inference__creator_336643"?
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
__inference__initializer_336651"?
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
__inference__destroyer_336656"?
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
__inference__creator_336661"?
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
__inference__initializer_336666"?
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
__inference__destroyer_336671"?
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
-:+2Adam/embedding_3/embeddings/m
%:#@2Adam/dense_4/kernel/m
:@2Adam/dense_4/bias/m
%:#@ 2Adam/dense_5/kernel/m
: 2Adam/dense_5/bias/m
R:P	?2AAdam/token_and_position_embedding_2/token_embedding3/embeddings/m
T:R 2DAdam/token_and_position_embedding_2/position_embedding3/embeddings/m
R:P2>Adam/transformer_encoder_3/multi_head_attention/query/kernel/m
L:J2<Adam/transformer_encoder_3/multi_head_attention/query/bias/m
P:N2<Adam/transformer_encoder_3/multi_head_attention/key/kernel/m
J:H2:Adam/transformer_encoder_3/multi_head_attention/key/bias/m
R:P2>Adam/transformer_encoder_3/multi_head_attention/value/kernel/m
L:J2<Adam/transformer_encoder_3/multi_head_attention/value/bias/m
]:[2IAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/m
S:Q2GAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/m
B:@26Adam/transformer_encoder_3/layer_normalization/gamma/m
A:?25Adam/transformer_encoder_3/layer_normalization/beta/m
D:B28Adam/transformer_encoder_3/layer_normalization_1/gamma/m
C:A27Adam/transformer_encoder_3/layer_normalization_1/beta/m
9:72)Adam/transformer_encoder_3/dense/kernel/m
3:12'Adam/transformer_encoder_3/dense/bias/m
;:92+Adam/transformer_encoder_3/dense_1/kernel/m
5:32)Adam/transformer_encoder_3/dense_1/bias/m
R:P2>Adam/transformer_decoder_3/multi_head_attention/query/kernel/m
L:J2<Adam/transformer_decoder_3/multi_head_attention/query/bias/m
P:N2<Adam/transformer_decoder_3/multi_head_attention/key/kernel/m
J:H2:Adam/transformer_decoder_3/multi_head_attention/key/bias/m
R:P2>Adam/transformer_decoder_3/multi_head_attention/value/kernel/m
L:J2<Adam/transformer_decoder_3/multi_head_attention/value/bias/m
]:[2IAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/m
S:Q2GAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/m
B:@26Adam/transformer_decoder_3/layer_normalization/gamma/m
A:?25Adam/transformer_decoder_3/layer_normalization/beta/m
D:B28Adam/transformer_decoder_3/layer_normalization_1/gamma/m
C:A27Adam/transformer_decoder_3/layer_normalization_1/beta/m
9:72)Adam/transformer_decoder_3/dense/kernel/m
3:12'Adam/transformer_decoder_3/dense/bias/m
;:92+Adam/transformer_decoder_3/dense_1/kernel/m
5:32)Adam/transformer_decoder_3/dense_1/bias/m
-:+2Adam/embedding_3/embeddings/v
%:#@2Adam/dense_4/kernel/v
:@2Adam/dense_4/bias/v
%:#@ 2Adam/dense_5/kernel/v
: 2Adam/dense_5/bias/v
R:P	?2AAdam/token_and_position_embedding_2/token_embedding3/embeddings/v
T:R 2DAdam/token_and_position_embedding_2/position_embedding3/embeddings/v
R:P2>Adam/transformer_encoder_3/multi_head_attention/query/kernel/v
L:J2<Adam/transformer_encoder_3/multi_head_attention/query/bias/v
P:N2<Adam/transformer_encoder_3/multi_head_attention/key/kernel/v
J:H2:Adam/transformer_encoder_3/multi_head_attention/key/bias/v
R:P2>Adam/transformer_encoder_3/multi_head_attention/value/kernel/v
L:J2<Adam/transformer_encoder_3/multi_head_attention/value/bias/v
]:[2IAdam/transformer_encoder_3/multi_head_attention/attention_output/kernel/v
S:Q2GAdam/transformer_encoder_3/multi_head_attention/attention_output/bias/v
B:@26Adam/transformer_encoder_3/layer_normalization/gamma/v
A:?25Adam/transformer_encoder_3/layer_normalization/beta/v
D:B28Adam/transformer_encoder_3/layer_normalization_1/gamma/v
C:A27Adam/transformer_encoder_3/layer_normalization_1/beta/v
9:72)Adam/transformer_encoder_3/dense/kernel/v
3:12'Adam/transformer_encoder_3/dense/bias/v
;:92+Adam/transformer_encoder_3/dense_1/kernel/v
5:32)Adam/transformer_encoder_3/dense_1/bias/v
R:P2>Adam/transformer_decoder_3/multi_head_attention/query/kernel/v
L:J2<Adam/transformer_decoder_3/multi_head_attention/query/bias/v
P:N2<Adam/transformer_decoder_3/multi_head_attention/key/kernel/v
J:H2:Adam/transformer_decoder_3/multi_head_attention/key/bias/v
R:P2>Adam/transformer_decoder_3/multi_head_attention/value/kernel/v
L:J2<Adam/transformer_decoder_3/multi_head_attention/value/bias/v
]:[2IAdam/transformer_decoder_3/multi_head_attention/attention_output/kernel/v
S:Q2GAdam/transformer_decoder_3/multi_head_attention/attention_output/bias/v
B:@26Adam/transformer_decoder_3/layer_normalization/gamma/v
A:?25Adam/transformer_decoder_3/layer_normalization/beta/v
D:B28Adam/transformer_decoder_3/layer_normalization_1/gamma/v
C:A27Adam/transformer_decoder_3/layer_normalization_1/beta/v
9:72)Adam/transformer_decoder_3/dense/kernel/v
3:12'Adam/transformer_decoder_3/dense/bias/v
;:92+Adam/transformer_decoder_3/dense_1/kernel/v
5:32)Adam/transformer_decoder_3/dense_1/bias/v
?B?
__inference_save_fn_336690checkpoint_key"?
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
__inference_restore_fn_336698restored_tensors_0restored_tensors_1"?
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
	?	
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant7
__inference__creator_336643?

? 
? "? 7
__inference__creator_336661?

? 
? "? 9
__inference__destroyer_336656?

? 
? "? 9
__inference__destroyer_336671?

? 
? "? C
__inference__initializer_336651 ????

? 
? "? ;
__inference__initializer_336666?

? 
? "? ?
!__inference__wrapped_model_332788?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\Z?W
P?M
K?H
 ?
Phrase?????????
$?!

Token_role????????? 
? "1?.
,
dense_5!?
dense_5????????? p
__inference_adapt_step_202704O??C?@
9?6
4?1?
??????????IteratorSpec 
? "
 ?
A__inference_add_2_layer_call_and_return_conditional_losses_335797?b?_
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
&__inference_add_2_layer_call_fn_335791?b?_
X?U
S?P
&?#
inputs/0????????? 
&?#
inputs/1????????? 
? "?????????? ?
C__inference_dense_4_layer_call_and_return_conditional_losses_336618\ST/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? {
(__inference_dense_4_layer_call_fn_336607OST/?,
%?"
 ?
inputs?????????
? "??????????@?
C__inference_dense_5_layer_call_and_return_conditional_losses_336638\[\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0????????? 
? {
(__inference_dense_5_layer_call_fn_336627O[\/?,
%?"
 ?
inputs?????????@
? "?????????? ?
G__inference_embedding_3_layer_call_and_return_conditional_losses_335785_&/?,
%?"
 ?
inputs????????? 
? ")?&
?
0????????? 
? ?
,__inference_embedding_3_layer_call_fn_335775R&/?,
%?"
 ?
inputs????????? 
? "?????????? ?
V__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_336598{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
;__inference_global_average_pooling1d_2_layer_call_fn_336592nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
C__inference_model_2_layer_call_and_return_conditional_losses_334491?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
X?U
K?H
 ?
Phrase?????????
$?!

Token_role????????? 
p 

 
? "%?"
?
0????????? 
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_334628?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
X?U
K?H
 ?
Phrase?????????
$?!

Token_role????????? 
p

 
? "%?"
?
0????????? 
? ?
C__inference_model_2_layer_call_and_return_conditional_losses_335301?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
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
C__inference_model_2_layer_call_and_return_conditional_losses_335732?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
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
(__inference_model_2_layer_call_fn_333404?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
X?U
K?H
 ?
Phrase?????????
$?!

Token_role????????? 
p 

 
? "?????????? ?
(__inference_model_2_layer_call_fn_334354?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
X?U
K?H
 ?
Phrase?????????
$?!

Token_role????????? 
p

 
? "?????????? ?
(__inference_model_2_layer_call_fn_334820?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
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
(__inference_model_2_layer_call_fn_334912?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\b?_
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
__inference_restore_fn_336698Z?K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_336690??&?#
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
$__inference_signature_wrapper_334728?/????]^&_`abcdefghklmnijopqrstuvwx{|}~yzST[\m?j
? 
c?`
*
Phrase ?
Phrase?????????
2

Token_role$?!

Token_role????????? "1?.
,
dense_5!?
dense_5????????? ?
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_335768`]^/?,
%?"
 ?
inputs????????? 	
? ")?&
?
0????????? 
? ?
?__inference_token_and_position_embedding_2_layer_call_fn_335741S]^/?,
%?"
 ?
inputs????????? 	
? "?????????? ?
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336393?opqrstuvwx{|}~yzU?R
K?H
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
p 
? ")?&
?
0????????? 
? ?
Q__inference_transformer_decoder_3_layer_call_and_return_conditional_losses_336587?opqrstuvwx{|}~yzU?R
K?H
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
p
? ")?&
?
0????????? 
? ?
6__inference_transformer_decoder_3_layer_call_fn_336183?opqrstuvwx{|}~yzU?R
K?H
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
p 
? "?????????? ?
6__inference_transformer_decoder_3_layer_call_fn_336220?opqrstuvwx{|}~yzU?R
K?H
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
p
? "?????????? ?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_335998~_`abcdefghklmnij??<
5?2
$?!
inputs????????? 

 

 
p 
? ")?&
?
0????????? 
? ?
Q__inference_transformer_encoder_3_layer_call_and_return_conditional_losses_336146~_`abcdefghklmnij??<
5?2
$?!
inputs????????? 

 

 
p
? ")?&
?
0????????? 
? ?
6__inference_transformer_encoder_3_layer_call_fn_335834q_`abcdefghklmnij??<
5?2
$?!
inputs????????? 

 

 
p 
? "?????????? ?
6__inference_transformer_encoder_3_layer_call_fn_335871q_`abcdefghklmnij??<
5?2
$?!
inputs????????? 

 

 
p
? "?????????? 