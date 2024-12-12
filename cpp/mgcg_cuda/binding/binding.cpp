#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "solve_soft.h"

namespace py = pybind11;

void addModule_SolveSoft(py::module &m);

PYBIND11_MODULE(pypbd, m) {
    addModule_SolveSoft(m);
}

void addModule_SolveSoft(py::module &m) {

    py::class_<SolveSoft>(m, "SolveSoft")
        .def(py::init<>())
        .def("solve", &SolveSoft::solve)
        .def("resize_fields", &SolveSoft::resize_fields)
        .def_readwrite("pos", &SolveSoft::pos)
        .def_readwrite("alpha_tilde", &SolveSoft::alpha_tilde)
        .def_readwrite("rest_len", &SolveSoft::rest_len)
        .def_readwrite("vert", &SolveSoft::vert)
        .def_readwrite("inv_mass", &SolveSoft::inv_mass)
        .def_readwrite("delta_t", &SolveSoft::delta_t)
        .def_readwrite("B", &SolveSoft::B)
        .def_readwrite("lambda", &SolveSoft::lambda)
        .def_readwrite("dlambda", &SolveSoft::dlambda)
        .def_readwrite("dpos", &SolveSoft::dpos)
        .def_readwrite("constraints", &SolveSoft::constraints)
        .def_readwrite("gradC", &SolveSoft::gradC)
        .def_readwrite("b", &SolveSoft::b)
        ;
}