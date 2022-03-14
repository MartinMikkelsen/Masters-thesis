using Printf, Dates

# functions to obtain neighbors of a given site i
up(neighs, i) = neighs[1, i]
right(neighs, i) = neighs[2, i]
down(neighs, i) = neighs[3, i]
left(neighs, i) = neighs[4, i]

function montecarlo(; L, T)
    # set parameters & initialize
    nsweeps = 10^7
    measure_rate = 5_000
    beta = 1/T
    conf = rand([-1, 1], L, L)
    confs = Matrix{Int64}[] # storing intermediate configurations
    # build nearest neighbor lookup table
    lattice = reshape(1:L^2, (L, L))
    ups     = circshift(lattice, (-1,0))
    rights  = circshift(lattice, (0,-1))
    downs   = circshift(lattice,(1,0))
    lefts   = circshift(lattice,(0,1))
    neighs = vcat(ups[:]',rights[:]',downs[:]',lefts[:]')
    
    start_time = now()
    println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
    
    # walk over the lattice and propose to flip each spin `nsweeps` times
    for i in 1:nsweeps
        # sweep
        for i in eachindex(conf)
            # calculate ΔE
            ΔE = 2.0 * conf[i] * (conf[up(neighs, i)] + conf[right(neighs, i)] +
                                + conf[down(neighs, i)] + conf[left(neighs, i)])
            # Metropolis criterium
            if ΔE <= 0 || rand() < exp(- beta*ΔE)
                conf[i] *= -1 # flip spin
            end
        end
        
        # store the spin configuration
        iszero(mod(i, measure_rate)) && push!(confs, copy(conf))
    end
    
    end_time = now()
    println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
    @printf("Duration: %.2f minutes", (end_time - start_time).value / 1000. /60.)
    
    # return the recorded spin configurations
    return confs
end