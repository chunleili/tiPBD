# function computesymbols(operator::Operator, θ::Array{<:Real})
#     # validity check
#     dimension = length(θ)
#     if dimension != operator.inputs[1].basis.dimension
#         throw(ArgumentError("Must provide as many values of θ as the mesh has dimensions")) # COV_EXCL_LINE
#     end

#     # setup
#     rowmodemap = operator.rowmodemap
#     columnmodemap = operator.columnmodemap
#     elementmatrix = operator.elementmatrix
#     numberrows, numbercolumns = size(elementmatrix)
#     nodecoordinatedifferences = operator.nodecoordinatedifferences
#     symbolmatrixnodes = zeros(ComplexF64, numberrows, numbercolumns)

#     # compute
#     for i = 1:numberrows, j = 1:numbercolumns
#         symbolmatrixnodes[i, j] =
#             elementmatrix[i, j] *
#             ℯ^(im * sum([θ[k] * nodecoordinatedifferences[i, j, k] for k = 1:dimension]))
#     end
#     symbolmatrixmodes = rowmodemap * symbolmatrixnodes * columnmodemap

#     # return
#     return symbolmatrixmodes
# end

import numpy as np

def computesymbols(operator, θ):
    dimension = len(θ)
    if dimension != operator.inputs[0].basis.dimension:
        raise ValueError("Must provide as many values of θ as the mesh has dimensions")
    
    rowmodemap = operator.rowmodemap
    columnmodemap = operator.columnmodemap
    elementmatrix = operator.elementmatrix
    numberrows, numbercolumns = elementmatrix.shape
    nodecoordinatedifferences = operator.nodecoordinatedifferences
    symbolmatrixnodes = np.zeros((numberrows, numbercolumns), dtype=np.complex128)

    for i in range(numberrows):
        for j in range(numbercolumns):
            symbolmatrixnodes[i, j] = elementmatrix[i, j] * np.exp(1j * sum([θ[k] * nodecoordinatedifferences[i, j, k] for k in range(dimension)])) 
    
    symbolmatrixmodes = np.dot(np.dot(rowmodemap, symbolmatrixnodes), columnmodemap)
    return symbolmatrixmodes



EvaluationMode = {
    'interpolation': 0,
    'gradient': 1,
    'quadratureweights': 2
}

class Operator:
    def __init__(self, weakform, mesh, inputs, outputs):
        self.weakform = weakform
        self.mesh = mesh
        self.inputs = inputs
        self.outputs = outputs
        self.dimension = 0
        self.numberquadraturepoints = 0

        if len(inputs) < 1:
            raise ValueError("must have at least one input")
        
        for input in inputs:
            if self.dimension == 0:
                self.dimension = input.basis.dimension
            if input.basis.dimension != self.dimension:
                raise ValueError("bases must have compatible dimensions")
            
            if self.numberquadraturepoints == 0:
                self.numberquadraturepoints = input.basis.numberquadraturepoints
            if input.basis.numberquadraturepoints != self.numberquadraturepoints:
                raise ValueError("bases must have compatible quadrature spaces")
        
        if len(outputs) < 1:
            raise ValueError("must have at least one output")
        
        for output in outputs:
            if EvaluationMode.quadratureweights in output.evaluationmodes:
                raise ValueError("quadrature weights is not a valid output")
            
            if output.basis.dimension != self.dimension:
                raise ValueError("bases must have compatible dimensions")
            
            if output.basis.numberquadraturepoints != self.numberquadraturepoints:
                raise ValueError("bases must have compatible quadrature spaces")
        
        if (self.dimension == 1 and type(mesh) != Mesh1D) or (self.dimension == 2 and type(mesh) != Mesh2D) or (self.dimension == 3 and type(mesh) != Mesh3D):
            raise ValueError("mesh dimension must match bases dimension")
        
        self.elementmatrix = None
        self.diagonal = None
        self.multiplicity = None
        self.rowmodemap = None
        self.columnmodemap = None
        self.inputcoordinates = None
        self.outputcoordinates = None
        self.nodecoordinatedifferences = None

# mutable struct Operator
#     # data never changed
#     weakform::Function
#     mesh::Mesh
#     inputs::AbstractArray{OperatorField}
#     outputs::AbstractArray{OperatorField}

#     # data empty until assembled
#     elementmatrix::AbstractArray{Float64,2}
#     diagonal::AbstractArray{Float64}
#     multiplicity::AbstractArray{Float64}
#     rowmodemap::AbstractArray{Float64,2}
#     columnmodemap::AbstractArray{Float64,2}
#     inputcoordinates::AbstractArray{Float64}
#     outputcoordinates::AbstractArray{Float64}
#     nodecoordinatedifferences::AbstractArray{Float64}

#     # inner constructor
#     Operator(
#         weakform::Function,
#         mesh::Mesh,
#         inputs::AbstractArray{OperatorField},
#         outputs::AbstractArray{OperatorField},
#     ) = (
#         dimension = 0;
#         numberquadraturepoints = 0;

#         # check inputs valididy
#         if length(inputs) < 1
#             error("must have at least one input") # COV_EXCL_LINE
#         end;
#         for input in inputs
#             # dimension
#             if dimension == 0
#                 dimension = input.basis.dimension
#             end
#             if input.basis.dimension != dimension
#                 error("bases must have compatible dimensions") # COV_EXCL_LINE
#             end

#             # number of quadrature points
#             if numberquadraturepoints == 0
#                 numberquadraturepoints = input.basis.numberquadraturepoints
#             end
#             if input.basis.numberquadraturepoints != numberquadraturepoints
#                 error("bases must have compatible quadrature spaces") # COV_EXCL_LINE
#             end
#         end;

#         # check outputs valididy
#         if length(outputs) < 1
#             error("must have at least one output") # COV_EXCL_LINE
#         end;
#         for output in outputs
#             # evaluation modes
#             if EvaluationMode.quadratureweights in output.evaluationmodes
#                 error("quadrature weights is not a valid output") # COV_EXCL_LINE
#             end

#             # dimension
#             if output.basis.dimension != dimension
#                 error("bases must have compatible dimensions") # COV_EXCL_LINE
#             end

#             # number of quadrature points
#             if output.basis.numberquadraturepoints != numberquadraturepoints
#                 error("bases must have compatible quadrature spaces") # COV_EXCL_LINE
#             end
#         end;

#         # check mesh valididy
#         if (dimension == 1 && typeof(mesh) != Mesh1D) ||
#            (dimension == 2 && typeof(mesh) != Mesh2D) ||
#            (dimension == 3 && typeof(mesh) != Mesh3D)
#             error("mesh dimension must match bases dimension") # COV_EXCL_LINE
#         end;

#         # constructor
#         new(weakform, mesh, inputs, outputs)
#     )
# end


class Mesh1D:
    def __init__(self, dx):
        if dx < 1e-14:
            raise ValueError("Mesh scaling must be positive")
        
        self.dimension = 1
        self.dx = dx
        self.volume = dx

class Mesh2D:
    def __init__(self, dx, dy):
        if dx < 1e-14 or dy < 1e-14:
            raise ValueError("Mesh scaling must be positive")
        
        self.dimension = 2
        self.dx = dx
        self.dy = dy
        self.volume = dx * dy


class Mesh3D:
    def __init__(self, dx, dy, dz):
        if dx < 1e-14 or dy < 1e-14 or dz < 1e-14:
            raise ValueError("Mesh scaling must be positive")
        
        self.dimension = 3
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.volume = dx * dy * dz