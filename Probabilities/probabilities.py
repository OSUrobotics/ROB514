#!/usr/bin/env python3

# ROB456 Homework 1
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Initialize random seed
np.random.seed(0)

# Problem I & II: Representing a pdf using a uniform set of bins


# Problem II: Use a polynomial f(x) = -0.1x^3 + 4x^2 - 0.1x + 10
def f_x(x_in):
    """
    This is just a made-up function. It could be anything... it's used to make a PDF
    :param x_in: x value - for this hwk, use x between -10 and 25
    :return: f(x)
    """

    # How numpy makes a polynomial function
    c = np.array([-0.1, 4.0, -0.1, 10.0], float)
    p = np.polyval(c, x_in)
    return p


def plot_pmf(ax, pmf=(0.1, 0.8, 0.1), x_vals=(-1, 0, 1), title='No title'):
    """
    Plot a pmf as a set of bars
    :param ax: Figure axes. If None, will call subplots
    :param pmf: An array of height values, set to [0.1, 0.8, 0.1] for default
    :param x_vals: x values to place bar plots at.
    :param title: Title of plot
    :return: Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    bar_width = [x_vals[i+1] - x_vals[i] for i in range(0, len(x_vals)-1)]

    # Just set the last width to be the average of the others
    bar_width.append(np.mean(bar_width))
    ax.bar(x_vals, pmf, width=bar_width, edgecolor='k')
    ax.set_title(title)
    return ax


def create_plot(pdf, hist=(0, 1, 0), pmf=(0, 1, 0), xlim_pf=(-1, 1), title='No title'):
    """
    Plot a pdf, a histogram, and a pmf. Assumes the x values of the hist/pmf are uniformly sampled from xlim_pf
    :param pdf: The probability density function (as a function). Default is f_x
    :param hist: A histogram. Default is [0,1,0]
    :param pmf: The probability mass function (as an array). Default is [0,1,0]
    :param xlim_pf: The left and right bounds for the pdf & pmf. Default is [-1,1]
    :param title: Title on the plot
    :return: None
    """
    nrows = 1
    ncols = 3

    f, (ax1, ax2, ax3) = plt.subplots(nrows, ncols)
    # Plot the probability distribution function
    xs = np.linspace(xlim_pf[0], xlim_pf[1])
    ys = [pdf(x) for x in xs]
    axs = [ax1, ax3]
    for i in [0, 1]:
        axs[i].plot(xs, ys, 'g-')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('prob')
        axs[i].set_title("{} PDF".format(title))

    xlim_hist = np.linspace(xlim_pf[0], xlim_pf[1], len(hist))
    # Plot the histogram
    plot_pmf(ax2, hist, xlim_hist, "{} hist".format(title))
    ax2.set_xlabel('x')
    ax2.set_ylabel('n samples')

    # The pmf and the pdf
    plot_pmf(ax3, pmf, xlim_hist, "{} pmf & pdf".format(title))
    ax3.set_xlabel('x')
    ax3.set_ylabel('prob')

    # The block=False keeps the plot from stopping after the first plot
    plt.show(block=False)


# A generic Gaussian - note that this is *not* a normalized probability distribution function
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


# Homework 1 problem 1:
# Plot a normalized gaussian, samples from that gaussain, and normalized samples
def plot_gauss_sampled(mu=0.0, sig=0.1, x_lim=(-1, 1), n=10):
    """
    Plot a Gaussian with the given parameters, then normalize and sample to create a pdf and a pmf
    :param mu: Center of the gaussian
    :param sig: Sigma - related to standard deviation/width
    :param x_lim: Start and stop values. Default is [-1,1]
    :param n: Number of samples to use
    :return: None
    """

    # begin homework 1 - Problem 1
    # Use a lambda function to create the unnormalized pdf
    # calculate area under curve for pdf
    # Create a normalized pdf by dividing by the area
    # calculate area under curve for normalized pdf (should be 1)
    # Sample to create the histogram
    # Sum of the histogram values
    # Create normalized histogram/pmf
    # Plot
    # Print area answers
    # end homework 1 - Problem 1


# Homework 1 problem 2:
# Create a PMF from the function f_x above. Use x between -10 and 25 on the function f_x, but
# shift everything so your final PDF/PMF is from -1 to 1
def plot_f_sampled(n=10):
    """
    Plot the f_x function above, then normalize and shift to 0,1 to create a pdf and a pmf
    :param n - number of samples
    :return: The PMF
    """

    # x limits we're using to evaluate f_x
    x_lim_fx = (-10, 25)

    # x limits we want to use for pdf
    x_lim_pdf = (-1, 1)

    # for the return - you should be creating this
    pmf = (0.1, 0.8, 0.1)

    # begin homework 1 - Problem 2
    # calculate area under curve for f_x
    # Create a normalized, shifted pdf by dividing by the area and shifting x
    # so x=-1 goes to -10, x=1 goes to 25
    # Create a function that maps x in x_lim_pdf to x in x_lim
    # Build a lambda function that uses the mapping function and normalizes for area
    # calculate area under curve for normalized pdf (should be 1)
    # Sample to create the histogram
    # Summed values of the histogram
    # Create normalized histogram/pmf
    # Plot
    # Print area answers
    # end homework 1 - Problem 2

    return pmf


# Homework 1 problem 3:
# Sample from a PMF.
def plot_pmf_samples(pmf=(0.1, 0.8, 0.1), x_lim=(0, 1), n=10):
    """
    Generate samples from the PMF then plot how many samples were found in each bin
    :param pmf: Probability mass function, as an iterable
    :param x_lim: Limits on x (2 numbers)
    :param n - number of samples
    :return: None
    """

    # Make the subplots
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot the pmf using uniform samples spaced in x
    x_vals = np.linspace(x_lim[0], x_lim[1], len(pmf))
    plot_pmf(ax1, pmf, x_vals, "PMF")

    # begin homework 1 - Problem 3
    # Generate uniform samples in the range of x_lim (use numpy's uniform function)
    # Make the boundaries of the bins by taking a running sum
    # Figure out which bin to put each sample in
    # plot the counts using plot_pmf
    # normalize the bin counts and plot
    # plot the normalized bin counts
    # end homework 1 - problem 3


if __name__ == '__main__':
    print("Begin homework 1, problem 1")
    plot_gauss_sampled(mu=0, sig=0.1)
    # begin homework 1
    # end homework 1

    print("\nBegin homework 1, problem 2")
    pmf_for_test_plot = plot_f_sampled(n=20)

    print("\nBegin homework 1, problem 3")
    plot_pmf_samples(pmf_for_test_plot, x_lim=(0, 1), n=200)
    
    end = input("Hit q to end")
