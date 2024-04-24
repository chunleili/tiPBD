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


class EvaluationMode:
    interpolation = 0
    gradient = 1
    quadratureweights = 2

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
                # self.numberquadraturepoints = 16
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




def gausslobatto(n):
    if n <= 1:
        raise ValueError("Lobatto undefined for n ≤ 1.")
    elif n == 2:
        return [-1.0, 1.0], [1.0, 1.0]
    elif n == 3:
        return [-1.0, 0.0, 1.0], [1.0 / 3, 4.0 / 3, 1.0 / 3]
    else:
        # x, w = gaussjacobi(n - 2, 1.0, 1.0)
        x,w = [-0.4472135954999579, 0.4472135954999579], [0.6666666666666666, 0.6666666666666666]
        for i in range(len(x)):
            w[i] = w[i] / (1 - x[i]**2)
        x.insert(0, -1.0)
        x.append(1.0)
        w.insert(0, 2 / (n * (n - 1)))
        w.append(2 / (n * (n - 1)))
        x, w = [-1.0, -0.4472135954999579, 0.4472135954999579, 1.0], [0.16666666666666666, 0.8333333333333333, 0.8333333333333333, 0.16666666666666666]
        return x, w


class TensorBasis:
    def __init__(self, numbernodes1d, numberquadraturepoints1d, numbercomponents, dimension, nodes1d, quadraturepoints1d, quadratureweights1d, interpolation1d, gradient1d):
        self.numbernodes1d = numbernodes1d
        self.numberquadraturepoints1d = numberquadraturepoints1d
        self.numbercomponents = numbercomponents
        self.dimension = dimension
        self.nodes1d = nodes1d
        self.quadraturepoints1d = quadraturepoints1d
        self.quadratureweights1d = quadratureweights1d
        self.interpolation1d = interpolation1d
        self.gradient1d = gradient1d
        self.numberquadraturepoints = numberquadraturepoints1d ** dimension

def TensorH1LagrangeBasis(numbernodes1d, numberquadraturepoints1d, numbercomponents, dimension):
    if numbernodes1d < 2:
        raise ValueError("numbernodes1d must be greater than or equal to 2")
    
    if numberquadraturepoints1d < 1:
        raise ValueError("numberquadraturepoints1d must be greater than or equal to 1")
    
    if dimension < 1 or dimension > 3:
        raise ValueError("only 1D, 2D, or 3D bases are supported")
    
    # get nodes, quadrature points, and weights
    # nodes1d, = gausslobatto(numbernodes1d)
    nodes1d = [-1.0, -0.4472135954999579, 0.4472135954999579, 1.0]
    quadraturepoints1d = []
    quadratureweights1d = []
    # if collocatedquadrature:
    if False:
        ...
        # quadraturepoints1d, quadratureweights1d = gausslobatto(numberquadraturepoints1d)
    else:
        # quadraturepoints1d, quadratureweights1d = gausslegendre(numberquadraturepoints1d)
        quadraturepoints1d, quadratureweights1d = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526], [0.34785484513745385, 0.6521451548625462, 0.6521451548625462, 0.34785484513745385]
    
    # build interpolation, gradient matrices
    # interpolation1d, gradient1d = buildinterpolationandgradient(nodes1d, quadraturepoints1d)
    interpolation1d = [[0.6299431661034454, 0.472558747113818, -0.14950343104607952, 0.04700151782881607], [-0.07069479527385582, 0.972976186258263, 0.13253992624542693, -0.03482131722983419], [-0.03482131722983419, 0.13253992624542696, 0.9729761862582628, -0.07069479527385582], [0.04700151782881607, -0.14950343104607955, 0.47255874711381796, 0.6299431661034455]]
    # gradient1d = [-2.341837415390958, 2.787944890537088, -0.6351041115519563, 0.18899663640582656; -0.5167021357255352, -0.48795249031352683, 1.3379050992756671, -0.3332504732366054; 0.33325047323660545, -1.3379050992756674, 0.4879524903135269, 0.5167021357255351; -0.1889966364058266, 0.6351041115519563, -2.7879448905370876, 2.3418374153909585]
    gradient1d = [[-2.341837415390958, 2.787944890537088, -0.6351041115519563, 0.18899663640582656], [-0.5167021357255352, -0.48795249031352683, 1.3379050992756671, -0.3332504732366054], [0.33325047323660545, -1.3379050992756674, 0.4879524903135269, 0.5167021357255351], [-0.1889966364058266, 0.6351041115519563, -2.7879448905370876, 2.3418374153909585]]

    # if not isnothing(mapping):
    #     _, gprime = mapping
    #     gradient1d /= gprime(quadraturepoints1d)
    #     nodes1d = transformquadrature(nodes1d, nothing, mapping)
    #     quadraturepoints1d, quadratureweights1d = transformquadrature(quadraturepoints1d, quadratureweights1d, mapping)
    
    # use basic constructor
    return TensorBasis(numbernodes1d, numberquadraturepoints1d, numbercomponents, dimension, nodes1d, quadraturepoints1d, quadratureweights1d, interpolation1d, gradient1d)


class OperatorField:
    def __init__(self, basis, evaluationmodes, name=None):
        if len(evaluationmodes) > 1 and EvaluationMode.quadratureweights in evaluationmodes:
            raise ValueError("quadrature weights must be a separate operator field")
        
        self.basis = basis
        self.evaluationmodes = evaluationmodes
        self.name = name

# https://github.com/jeremylt/LFAToolkit.jl/blob/68d6bce58e4f46c2a81090e10b296b743a351ed9/src/Operator/Base.jl#L278
def getelementmatrix(operator):
    
def getrowmodemap(operator):
    if not hasattr(operator, 'rowmodemap'):
        numbermodes = 0
        for output in operator.outputs:
            if output.evaluationmodes[0] != EvaluationMode.quadratureweights:
                numbermodes += output.basis.numbernodes
        
        numbercolumns = operator.elementmatrix.shape[1]
        rowmodemap = np.zeros((numbermodes, numbercolumns))
        currentnode = 0
        currentmode = 0
        for output in operator.outputs:
            if output.evaluationmodes[0] != EvaluationMode.quadratureweights:
                for i in range(output.basis.numbernodes * output.basis.numbercomponents):
                    rowmodemap[output.basis.numbernodes * output.basis.numbercomponents + currentmode, i + currentnode] = 1
                
                currentnode += output.basis.numbernodes
                currentmode += output.basis.numbernodes
        
        operator.rowmodemap = rowmodemap
    
    return operator.rowmodemap


# ------------------------------------------------------------------------------
# mass matrix example
# ------------------------------------------------------------------------------

mesh = Mesh2D(1.0, 1.0)
basis = TensorH1LagrangeBasis(4, 4, 1, 2)

def massweakform(u, w):
    v = u * w[0]
    return [v]

inputs = [
    OperatorField(basis, [EvaluationMode.interpolation]),
    OperatorField(basis, [EvaluationMode.quadratureweights]),
]

outputs = [OperatorField(basis, [EvaluationMode.interpolation])]
mass = Operator(massweakform, mesh, inputs, outputs)

A = computesymbols(mass, [np.pi, np.pi])
eigenvalues = np.real(np.linalg.eigvals(A))
# ------------------------------------------------------------------------------
