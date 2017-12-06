import os
import numpy
import pyopencl as cl


class AppearanceCL(object):

    def __init__(self, lambda_occ, esim, appearance_norm_weight):
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
        self.cl_context = cl.create_some_context(False)
        self.queue = cl.CommandQueue(self.cl_context)
        program_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "patchmatch.cl")
        self.load_program(program_path)
        self.lambda_occ = lambda_occ
        self.esim = esim
        self.appearance_norm_weight = appearance_norm_weight

        self.energy = None
        self.images = None
        self.target_size = None
        self.source_size = None
        self.patch_size = None
        self.effective_target_size = None
        self.effective_source_size = None
        self.nff = None
        self.occurrence_map = None
        self.nff_buf = None
        self.const_occ = None
        self.images_buf = None
        self.occurrence_map_buf = None
        self.iteration = None
        self.energy_buf = None
        self.gradient_buf = None

    def load_program(self, filename):
        program_file = open(filename, 'r')
        program_text = "".join(program_file.readlines())
        self.program = cl.Program(self.cl_context, program_text).build()
        program_file.close()

    def random_fill(self):
        self.program.random_fill(
            self.queue, self.effective_target_size, None, self.nff_buf,
            numpy.int32(self.effective_source_size[0]),
            numpy.int32(self.effective_source_size[1]),
            numpy.int32(self.effective_target_size[1]))

    def initialize_distance(self):
        self.program.initialize_distance(
            self.queue, self.effective_target_size, None, self.images_buf[0],
            self.images_buf[1], self.nff_buf, self.occurrence_map_buf,
            numpy.int32(self.patch_size[0]), numpy.int32(self.patch_size[1]),
            numpy.int32(self.target_size[1]), numpy.int32(self.source_size[1]),
            numpy.int32(self.effective_source_size[0]),
            numpy.int32(self.effective_source_size[1]),
            numpy.int32(self.effective_target_size[1]),
            numpy.double(self.lambda_occ), numpy.double(self.esim),
            numpy.double(self.const_occ))

    def propagate(self):
        self.program.propagate(
            self.queue, self.effective_target_size, None, self.images_buf[0],
            self.images_buf[1], self.nff_buf, self.occurrence_map_buf,
            numpy.int32(self.patch_size[0]), numpy.int32(self.patch_size[1]),
            numpy.int32(self.target_size[0]), numpy.int32(self.target_size[1]),
            numpy.int32(self.source_size[0]), numpy.int32(self.source_size[1]),
            numpy.int32(self.effective_target_size[0]),
            numpy.int32(self.effective_target_size[1]),
            numpy.int32(self.effective_source_size[0]),
            numpy.int32(self.effective_source_size[1]),
            numpy.int32(self.iteration), numpy.double(self.lambda_occ),
            numpy.double(self.esim), numpy.double(self.const_occ))

    def build_occurence_map(self):
        self.program.build_occurrence_map(
            self.queue, self.effective_target_size, None,
            self.occurrence_map_buf, self.nff_buf,
            numpy.int32(self.patch_size[0]), numpy.int32(self.patch_size[1]),
            numpy.int32(self.source_size[1]),
            numpy.int32(self.effective_target_size[1]))

    def build_gradient(self):
        self.program.build_gradient(
            self.queue, self.target_size[0:2], None, self.images_buf[0],
            self.images_buf[1], self.nff_buf, self.energy_buf,
            self.gradient_buf, numpy.int32(self.patch_size[0]),
            numpy.int32(self.patch_size[1]), numpy.int32(self.target_size[1]),
            numpy.int32(self.source_size[1]),
            numpy.int32(self.effective_target_size[0]),
            numpy.int32(self.effective_target_size[1]),
            numpy.int32(self.effective_source_size[0]),
            numpy.int32(self.effective_source_size[1]),
            numpy.double(self.esim))

    def compute(self, target, source, gradient, patch_size, iterations):
        self.energy = numpy.zeros(1)
        self.images = [target, source]
        self.target_size = self.images[0].shape
        self.source_size = self.images[1].shape
        self.patch_size = patch_size
        self.effective_target_size = [self.target_size[i] - patch_size[i] + 1
                                      for i in (0, 1)]
        self.effective_source_size = [self.source_size[i] - patch_size[i] + 1
                                      for i in (0, 1)]
        assert all(x > 0 for x in self.effective_target_size), "Target dimensions too smalls."
        assert all(x > 0 for x in self.effective_source_size), "Source dimensions too smalls."
        self.nff = numpy.ndarray(
            (self.effective_target_size[0], self.effective_target_size[1], 3))
        self.occurrence_map = numpy.zeros(
            (self.source_size[0], self.source_size[1]), dtype=int)
        source_pixels = self.source_size[0] * self.source_size[1]
        target_pixels = self.target_size[0] * self.target_size[1]
        patch_pixels = self.patch_size[0] * self.patch_size[1]
        self.const_occ = source_pixels / (target_pixels * (patch_pixels ** 2))
        # neighborhood matching (patchmatch)
        self.nff_buf = cl.Buffer(self.cl_context, cl.mem_flags.READ_WRITE |
                                 cl.mem_flags.COPY_HOST_PTR, hostbuf=self.nff)
        self.images_buf = [cl.Buffer(self.cl_context, cl.mem_flags.READ_ONLY |
                                     cl.mem_flags.COPY_HOST_PTR, hostbuf=self.images[i])
                           for i in [0, 1]]
        self.occurrence_map_buf = cl.Buffer(
            self.cl_context, cl.mem_flags.READ_WRITE |
            cl.mem_flags.COPY_HOST_PTR, hostbuf=self.occurrence_map)
        self.random_fill()
        if self.lambda_occ > 0:
            self.build_occurence_map()
        self.initialize_distance()
        for i in range(iterations):
            self.iteration = i + 1
            self.propagate()
            if self.lambda_occ > 0:
                self.build_occurence_map()
        # appearance gradient
        self.energy_buf = cl.Buffer(
            self.cl_context, cl.mem_flags.WRITE_ONLY |
            cl.mem_flags.COPY_HOST_PTR, hostbuf=self.energy)
        if gradient is not None:
            self.gradient_buf = cl.Buffer(
                self.cl_context, cl.mem_flags.WRITE_ONLY |
                cl.mem_flags.COPY_HOST_PTR, hostbuf=gradient)
            self.build_gradient()
            cl.enqueue_read_buffer(self.queue, self.gradient_buf, gradient).wait()
        cl.enqueue_read_buffer(self.queue, self.energy_buf, self.energy).wait()

        # Experimental: appearance energy normalization (better convergence)
        if self.appearance_norm_weight > 0:
            norm_term = (self.effective_target_size[0] * self.effective_target_size[1] *
                         self.patch_size[0] * self.patch_size[1]) / self.appearance_norm_weight
            if gradient is not None:
                gradient[:] /= norm_term
            self.energy[0] /= norm_term

        return self.energy[0]
