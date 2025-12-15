#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <memory>

#include "csaps.h"

namespace csaps
{

DoubleArray Diff(const DoubleArray &vec)
{
  // behave like numpy.diff: if length < 2 -> empty
  if (vec.size() < 2) return DoubleArray();
  const Size n = vec.size() - 1;
  return vec.tail(n) - vec.head(n);
}

IndexArray Digitize(const DoubleArray &arr, const DoubleArray &bins)
{
  // This code works if `arr` and `bins` are monotonically increasing
  // returns indices in 1..bins.size()-1 such that bins[i-1] <= a < bins[i]
  IndexArray indexes = IndexArray::Ones(arr.size());

  auto IsInsideBin = [&bins](double a, Index index)
  {
    double bl = bins(index - 1);
    double br = bins(index);

    // Handle infinity cases specially to avoid tol=infinity
    bool lower_ok = false;
    bool upper_ok = false;

    if (std::isinf(bl)) {
      // -infinity case: a >= -inf is always true
      lower_ok = true;
    } else {
      // relative tolerance based on magnitude
      const double rel_prc = 1.e-8;
      double tol = rel_prc * std::max({1.0, std::abs(a), std::abs(bl)});
      lower_ok = (a >= bl - tol);
    }

    if (std::isinf(br)) {
      // +infinity case: a < +inf is always true
      upper_ok = true;
    } else {
      // relative tolerance based on magnitude
      const double rel_prc = 1.e-8;
      double tol = rel_prc * std::max({1.0, std::abs(a), std::abs(br)});
      upper_ok = (a < br + tol);
    }

    return lower_ok && upper_ok;
  };

  Index kstart = 1;

  for (Index i = 0; i < arr.size(); ++i) {
    double a = arr(i);
    // start from previous k to keep O(N) behavior for sorted arrays
    for (Index k = kstart; k < bins.size(); ++k) {
      if (IsInsideBin(a, k)) {
        indexes(i) = k;
        kstart = k;
        break;
      }
    }
  }

  return indexes;
}

DoubleSparseMatrix MakeSparseDiagMatrix(const DoubleArray2D& diags, const IndexArray& offsets, Size rows, Size cols)
{
  if (rows <= 0 || cols <= 0) {
    return DoubleSparseMatrix(std::max<Size>(0, rows), std::max<Size>(0, cols));  // Return empty sparse matrix
  }

  auto GetNumElemsAndIndex = [rows, cols](Index offset, Index &i, Index &j)
  {
    if (offset < 0) {
      i = -offset;
      j = 0;
    }
    else {
      i = 0;
      j = offset;
    }

    return std::min(rows - i, cols - j);
  };

  DoubleSparseMatrix m(rows, cols);

  for (Index k = 0; k < offsets.size(); ++k) {
    Index offset = offsets(k);
    Index i, j;

    Index n = GetNumElemsAndIndex(offset, i, j);
    if (n <= 0) continue;

    // diags.row(k) might be longer than n; decide which segment to use.
    // We follow the original idea: choose head or tail depending on rows vs cols and offset sign.
    // Implement explicitly to avoid ambiguous head()/tail() semantics for different sizes.
    const Index diag_len = static_cast<Index>(diags.cols());
    Index start_in_row = 0;

    if (offset < 0) {
      if (rows >= cols) {
        // use head(n): start 0
        start_in_row = 0;
      } else {
        // use tail(n)
        start_in_row = diag_len - n;
      }
    } else {
      if (rows >= cols) {
        // use tail(n)
        start_in_row = diag_len - n;
      } else {
        // use head(n)
        start_in_row = 0;
      }
    }

    for (Index l = 0; l < n; ++l) {
      // take element from diags.row(k) at start_in_row + l
      double val = diags.row(k)(start_in_row + l);
      m.insert(i + l, j + l) = val;
    }
  }

  m.makeCompressed();
  return m;
}

csaps::DoubleArray SolveLinearSystem(const DoubleSparseMatrix &A, const DoubleArray &b)
{
  Eigen::SparseLU<DoubleSparseMatrix> solver;

  // Compute the ordering permutation vector from the structural pattern of A
  solver.analyzePattern(A);

  // Compute the numerical factorization
  solver.factorize(A);

  // Use the factors to solve the linear system
  DoubleArray x = solver.solve(b.matrix()).array();

  return x;
}

double NormalizeSmooth(const DoubleArray &x, const DoubleArray &w, double smooth)
{
  // Normalize smoothing parameter based on data characteristics
  // See: https://github.com/espdev/csaps/pull/47

  // span: peak-to-peak range of x
  double span = x.maxCoeff() - x.minCoeff();

  // eff_x: effective x spacing
  DoubleArray dx = Diff(x);
  double dx_sq_sum = (dx * dx).sum();
  double eff_x = 1.0 + (span * span) / dx_sq_sum;

  // eff_w: effective weight
  double w_sum = w.sum();
  double w_sq_sum = (w * w).sum();
  double eff_w = (w_sum * w_sum) / w_sq_sum;

  // k factor
  double span_cubed = span * span * span;
  double n = x.size();
  double k = 80.0 * span_cubed / (n * n) / std::sqrt(eff_x) / std::sqrt(eff_w);

  // Normalize smooth parameter
  double s = (smooth < 0.0) ? 0.5 : smooth;
  double p = s / (s + (1.0 - s) * k);

  return p;
}

/* ---------------- Univariate spline implementation ---------------- */

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), -1.0)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights)
  : UnivariateCubicSmoothingSpline(xdata, ydata, weights, -1.0)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), smooth)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth)
  : m_xdata(xdata)
  , m_ydata(ydata)
  , m_weights(weights)
  , m_smooth(smooth)
{
  if (m_xdata.size() < 2) {
    throw std::runtime_error("There must be at least 2 data points");
  }

  if (m_weights.size() == 0) {
    m_weights = DoubleArray::Constant(m_xdata.size(), 1.0);
  }

  if (m_smooth > 1.0) {
    throw std::runtime_error("Smoothing parameter must be less than or equal 1.0");
  }

  if (m_xdata.size() != m_ydata.size() || m_xdata.size() != m_weights.size()) {
    throw std::runtime_error("Lenghts of the input data vectors are not equal");
  }

  MakeSpline();
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, double smooth, bool normalizedsmooth)
  : UnivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), smooth, normalizedsmooth)
{
}

UnivariateCubicSmoothingSpline::UnivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray &ydata, const DoubleArray &weights, double smooth, bool normalizedsmooth)
  : m_xdata(xdata)
  , m_ydata(ydata)
  , m_weights(weights)
  , m_smooth(smooth)
{
  if (m_xdata.size() < 2) {
    throw std::runtime_error("There must be at least 2 data points");
  }

  if (m_weights.size() == 0) {
    m_weights = DoubleArray::Constant(m_xdata.size(), 1.0);
  }

  if (m_smooth > 1.0 && m_smooth > 0.0) {  // Allow negative values for auto-smooth
    throw std::runtime_error("Smoothing parameter must be less than or equal 1.0");
  }

  if (m_xdata.size() != m_ydata.size() || m_xdata.size() != m_weights.size()) {
    throw std::runtime_error("Lenghts of the input data vectors are not equal");
  }

  MakeSpline(normalizedsmooth);
}

DoubleArray UnivariateCubicSmoothingSpline::operator()(const DoubleArray &xidata, int nu, bool extrapolate)
{
  if (xidata.size() < 2) {
    throw std::runtime_error("There must be at least 2 data points");
  }

  return Evaluate(xidata, nu, extrapolate);
}

DoubleArray UnivariateCubicSmoothingSpline::operator()(const Size pcount, DoubleArray &xidata, int nu, bool extrapolate)
{
  if (pcount < 2) {
    throw std::runtime_error("There must be at least 2 data points");
  }

  xidata.resize(pcount);
  xidata << DoubleArray::LinSpaced(pcount, m_xdata(0), m_xdata(m_xdata.size()-1));

  return Evaluate(xidata, nu, extrapolate);
}

static double safe_trace(const DoubleSparseMatrix &m)
{
  // compute trace safely
  double s = 0.0;
  for (int k = 0; k < m.outerSize(); ++k) {
    for (typename DoubleSparseMatrix::InnerIterator it(m, k); it; ++it) {
      if (it.row() == it.col()) s += it.value();
    }
  }
  return s;
}

void UnivariateCubicSmoothingSpline::MakeSpline(bool normalizedsmooth)
{
  const Size pcount = m_xdata.size();
  const Size pcount_m1 = pcount - 1;
  const Size pcount_m2 = pcount - 2;

  DoubleArray dx = Diff(m_xdata);
  DoubleArray dy = Diff(m_ydata);
  DoubleArray divdydx = dy / dx;

  double p = m_smooth;

  if (pcount > 2) {
    // Create diagonal sparse matrices
    const Size n = dx.size() - 1;

    DoubleArray2D diags(3, n);

    DoubleArray head_r = dx.head(n);
    DoubleArray tail_r = dx.tail(n);

    diags.row(0) = tail_r;
    diags.row(1) = 2 * (tail_r + head_r);
    diags.row(2) = head_r;

    IndexArray offsets(3);

    offsets << -1, 0, 1;

    DoubleSparseMatrix r = MakeSparseDiagMatrix(diags, offsets, pcount_m2, pcount_m2);

    DoubleArray odx = 1. / dx;

    DoubleArray head_qt = odx.head(n);
    DoubleArray tail_qt = odx.tail(n);

    diags.row(0) = head_qt;
    diags.row(1) = -(tail_qt + head_qt);
    diags.row(2) = tail_qt;

    offsets = IndexArray(3);
    offsets << 0, 1, 2;

    DoubleSparseMatrix qt = MakeSparseDiagMatrix(diags, offsets, pcount_m2, pcount);

    // Build w and qw as diagonal sparse matrices directly (safe and explicit)
    DoubleSparseMatrix w(pcount, pcount);
    DoubleSparseMatrix qw(pcount, pcount);
    w.reserve(Eigen::VectorXi::Constant(static_cast<int>(pcount), 1));
    qw.reserve(Eigen::VectorXi::Constant(static_cast<int>(pcount), 1));

    DoubleArray ow = 1. / m_weights;
    DoubleArray osqw = 1. / m_weights.sqrt();
    for (Index i = 0; i < pcount; ++i) {
      w.insert(i, i) = ow(i);
      qw.insert(i, i) = osqw(i);
    }
    w.makeCompressed();
    qw.makeCompressed();

    DoubleSparseMatrix qtw = qt * qw;
    DoubleSparseMatrix qtwq = qtw * qtw.transpose();

    auto Trace = [](const DoubleSparseMatrix &m)
    {
      // use diagonal sum - but use safe utility
      return safe_trace(m);
    };

    // Handle smooth parameter
    if (normalizedsmooth) {
      // Use normalized smooth
      p = NormalizeSmooth(m_xdata, m_weights, p);
    } else if (p < 0) {
      // Use automatic smooth calculation
      double tr_r = Trace(r);
      double tr_q = Trace(qtwq);
      if (tr_q == 0.0) {
        p = 1.0;
      } else {
        p = 1. / (1. + tr_r / (6. * tr_q));
      }
    }
    
    DoubleSparseMatrix A = ((6. * (1. - p)) * qtwq) + (p * r);
    A.makeCompressed();

    DoubleArray b = Diff(divdydx);

    // Solve linear system Ab = u
    DoubleArray u = SolveLinearSystem(A, b);
    
    DoubleArray d1 = DoubleArray::Zero(u.size() + 2);
    d1.segment(1, u.size()) = u;
    d1 = Diff(d1) / dx;

    DoubleArray d2 = DoubleArray::Zero(d1.size() + 2);
    d2.segment(1, d1.size()) = d1;
    d2 = Diff(d2);

    DoubleArray yi = m_ydata - ((6. * (1. - p)) * (w * d2.matrix()).array());
    
    DoubleArray c3 = DoubleArray::Zero(u.size() + 2);
    c3.segment(1, u.size()) = p * u;

    DoubleArray c2 = Diff(yi) / dx - dx * (2. * c3.head(pcount_m1) + c3.tail(pcount_m1));

    m_coeffs.resize(pcount_m1, 4);

    m_coeffs.col(0) = Diff(c3) / dx;
    m_coeffs.col(1) = 3. * c3.head(pcount_m1);
    m_coeffs.col(2) = c2;
    m_coeffs.col(3) = yi.head(pcount_m1);
  }
  else {
    p = 1.0;
    m_coeffs.resize(1, 2);
    m_coeffs(0, 0) = divdydx(0);
    m_coeffs(0, 1) = m_ydata(0);
  }

  m_smooth = p;
}

DoubleArray UnivariateCubicSmoothingSpline::Evaluate(const DoubleArray &xidata, int nu, bool extrapolate)
{
  const Size x_size = m_xdata.size();
  const Size xi_size = xidata.size();
  const Index max_piece = std::max<Index>(0, static_cast<Index>(m_coeffs.rows()) - 1);
  const bool is_linear = (m_coeffs.cols() == 2);

  // Prepare edges for digitization: [-inf, x1, x2, ..., +inf]
  DoubleArray edges(x_size);
  edges(0) = -DoubleLimits::infinity();
  if (x_size >= 3) {
    edges.segment(1, x_size - 2) = m_xdata.segment(1, x_size - 2);
  }
  edges(x_size - 1) = DoubleLimits::infinity();

  IndexArray indexes = Digitize(xidata, edges);
  indexes -= 1;  // Convert to 0-based indexing

  DoubleArray yidata(xi_size);

  if (is_linear && m_coeffs.cols() == 2) {
    // Linear spline: y = m*x + b
    for (Index k = 0; k < xi_size; ++k) {
      bool outside = (xidata(k) < m_xdata(0) || xidata(k) > m_xdata(m_xdata.size() - 1));
      if (outside && !extrapolate) {
        yidata(k) = DoubleLimits::quiet_NaN();
        continue;
      }

      Index idx = std::clamp<Index>(indexes(k), 0, max_piece);
      double dx = xidata(k) - m_xdata(idx);
      yidata(k) = dx * m_coeffs(idx, 0) + m_coeffs(idx, 1);
    }
  } else {
    // Cubic or higher order spline
    if (nu == 0) {
      // Function value: Horner evaluation
      for (Index k = 0; k < xi_size; ++k) {
        bool outside = (xidata(k) < m_xdata(0) || xidata(k) > m_xdata(m_xdata.size() - 1));
        if (outside && !extrapolate) {
          yidata(k) = DoubleLimits::quiet_NaN();
          continue;
        }

        Index idx = std::clamp<Index>(indexes(k), 0, max_piece);
        double dx = xidata(k) - m_xdata(idx);

        // Horner's method: start from highest degree
        double result = m_coeffs(idx, 0);
        for (Index col = 1; col < m_coeffs.cols(); ++col) {
          result = result * dx + m_coeffs(idx, col);
        }
        yidata(k) = result;
      }
    } else if (nu == 1) {
      // First derivative
      yidata = DoubleArray::Zero(xi_size);
      for (Index k = 0; k < xi_size; ++k) {
        bool outside = (xidata(k) < m_xdata(0) || xidata(k) > m_xdata(m_xdata.size() - 1));
        if (outside && !extrapolate) {
          yidata(k) = DoubleLimits::quiet_NaN();
          continue;
        }

        Index idx = std::clamp<Index>(indexes(k), 0, max_piece);
        double dx = xidata(k) - m_xdata(idx);

        // d/dx[c0*x^3 + c1*x^2 + c2*x + c3] = 3*c0*x^2 + 2*c1*x + c2
        double result = 0.0;
        Index power = m_coeffs.cols() - 1;  // Highest power in original polynomial
        for (Index col = 0; col < m_coeffs.cols() - 1; ++col) {
          result = result * dx + m_coeffs(idx, col) * static_cast<double>(power);
          power--;
        }
        yidata(k) = result;
      }
    } else if (nu == 2) {
      // Second derivative: d²/dx²[c0*x^3 + c1*x^2 + c2*x + c3] = 6*c0*x + 2*c1
      yidata = DoubleArray::Zero(xi_size);
      if (m_coeffs.cols() > 2) {
        for (Index k = 0; k < xi_size; ++k) {
          bool outside = (xidata(k) < m_xdata(0) || xidata(k) > m_xdata(m_xdata.size() - 1));
          if (outside && !extrapolate) {
            yidata(k) = DoubleLimits::quiet_NaN();
            continue;
          }

          Index idx = std::clamp<Index>(indexes(k), 0, max_piece);
          double dx = xidata(k) - m_xdata(idx);

          // Horner evaluation: 6*c0*x + 2*c1
          double result = m_coeffs(idx, 0) * 6.0;
          result = result * dx + m_coeffs(idx, 1) * 2.0;
          yidata(k) = result;
        }
      }
    } else if (nu == 3) {
      // Third derivative
      yidata = DoubleArray::Zero(xi_size);
      if (m_coeffs.cols() > 3) {
        for (Index k = 0; k < xi_size; ++k) {
          bool outside = (xidata(k) < m_xdata(0) || xidata(k) > m_xdata(m_xdata.size() - 1));
          if (outside && !extrapolate) {
            yidata(k) = DoubleLimits::quiet_NaN();
            continue;
          }

          Index idx = std::clamp<Index>(indexes(k), 0, max_piece);
          // d³/dx³[c0*x^3 + c1*x^2 + c2*x + c3] = 6*c0
          yidata(k) = m_coeffs(idx, 0) * 6.0;
        }
      }
    } else {
      // Higher derivatives are zero
      yidata = DoubleArray::Zero(xi_size);
    }
  }

  return yidata;
}

/* ---------------- Multivariate implementation ---------------- */

MultivariateCubicSmoothingSpline::MultivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray2D &ydata)
  : MultivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), -1.0)
{}

MultivariateCubicSmoothingSpline::MultivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray2D &ydata, const DoubleArray &weights)
  : MultivariateCubicSmoothingSpline(xdata, ydata, weights, -1.0)
{}

MultivariateCubicSmoothingSpline::MultivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray2D &ydata, double smooth)
  : MultivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), smooth)
{}

MultivariateCubicSmoothingSpline::MultivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray2D &ydata, const DoubleArray &weights, double smooth)
  : m_xdata(xdata), m_ydata(ydata), m_weights(weights), m_smooth(smooth)
{
  if (m_xdata.size() < 2) throw std::runtime_error("There must be at least 2 xdata points");
  if (m_ydata.rows() != m_xdata.size()) throw std::runtime_error("ydata rows must equal xdata.size()");
  m_dims = static_cast<Index>(m_ydata.cols());

  if (m_weights.size() == 0) {
    m_weights = DoubleArray::Constant(m_xdata.size(), 1.0);
  } else {
    if (m_weights.size() != m_xdata.size()) throw std::runtime_error("weights length must equal xdata length");
  }

  MakeSplines();
}

MultivariateCubicSmoothingSpline::MultivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray2D &ydata, double smooth, bool normalizedsmooth)
  : MultivariateCubicSmoothingSpline(xdata, ydata, DoubleArray(), smooth, normalizedsmooth)
{}

MultivariateCubicSmoothingSpline::MultivariateCubicSmoothingSpline(const DoubleArray &xdata, const DoubleArray2D &ydata, const DoubleArray &weights, double smooth, bool normalizedsmooth)
  : m_xdata(xdata), m_ydata(ydata), m_weights(weights), m_smooth(smooth)
{
  if (m_xdata.size() < 2) throw std::runtime_error("There must be at least 2 xdata points");
  if (m_ydata.rows() != m_xdata.size()) throw std::runtime_error("ydata rows must equal xdata.size()");
  m_dims = static_cast<Index>(m_ydata.cols());

  if (m_weights.size() == 0) {
    m_weights = DoubleArray::Constant(m_xdata.size(), 1.0);
  } else {
    if (m_weights.size() != m_xdata.size()) throw std::runtime_error("weights length must equal xdata length");
  }

  MakeSplines(normalizedsmooth);
}

void MultivariateCubicSmoothingSpline::MakeSplines(bool normalizedsmooth)
{
  m_splines.clear();
  m_splines.resize(static_cast<size_t>(m_dims));
  m_smooths = DoubleArray::Constant(m_dims, 0.0);

  // Build per-dimension splines. Possibly parallelize.
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (Index d = 0; d < m_dims; ++d) {
    // Extract column d as DoubleArray
    DoubleArray col = m_ydata.col(d);
    // When m_smooth >= 0, pass it; otherwise pass -1.0 to request auto-smooth
    double s = m_smooth;
    // construct spline (unique_ptr) and store
    std::unique_ptr<UnivariateCubicSmoothingSpline> sp;
    try {
      if (normalizedsmooth) {
        if (s >= 0.0) {
          sp = std::make_unique<UnivariateCubicSmoothingSpline>(m_xdata, col, m_weights, s, true);
        } else {
          sp = std::make_unique<UnivariateCubicSmoothingSpline>(m_xdata, col, m_weights, -1.0, true);
        }
      } else {
        if (s >= 0.0) {
          sp = std::make_unique<UnivariateCubicSmoothingSpline>(m_xdata, col, m_weights, s);
        } else {
          sp = std::make_unique<UnivariateCubicSmoothingSpline>(m_xdata, col, m_weights);
        }
      }
    } catch (const std::exception &e) {
      // rethrow with more context
      std::ostringstream oss;
      oss << "Failed to construct UnivariateCubicSmoothingSpline for dim " << d << " : " << e.what();
      throw std::runtime_error(oss.str());
    } catch (...) {
      std::ostringstream oss;
      oss << "Failed to construct UnivariateCubicSmoothingSpline for dim " << d;
      throw std::runtime_error(oss.str());
    }

    // store
    m_smooths(d) = sp->GetSmooth();
    m_splines[static_cast<size_t>(d)] = std::move(sp);
  }
}

DoubleArray2D MultivariateCubicSmoothingSpline::operator()(const DoubleArray &xidata, int nu, bool extrapolate)
{
  if (xidata.size() < 1) return DoubleArray2D(0, 0);

  const Index nx = xidata.size();
  const Index dims = m_dims;
  DoubleArray2D out(nx, dims);

  // Evaluate each spline on xidata and place into column d
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for (Index d = 0; d < dims; ++d) {
    auto &sp = m_splines[static_cast<size_t>(d)];
    if (!sp) {
      // shouldn't happen
      for (Index i = 0; i < nx; ++i) out(i, d) = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    DoubleArray col = (*sp)(xidata, nu, extrapolate); // returns ArrayXd of length nx with derivative support
    // copy to out column
    for (Index i = 0; i < nx; ++i) out(i, d) = col(i);
  }

  return out;
}

DoubleArray MultivariateCubicSmoothingSpline::GetSmooths() const
{
  return m_smooths;
}

} // namespace csaps
