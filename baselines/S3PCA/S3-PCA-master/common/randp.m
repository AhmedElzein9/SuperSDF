function R = randp(P, varargin)
% RANDP pseudorandom integers from a specified discrete distribution
%    R = RANDP(P, N) returns an N-by-N matrix containing pseudorandom
%    integers drawn from a specified discrete distribution on 1:numel(P).
%    The distribution is specified by the relative values of P so that a
%    value K is present approximately "P(K)/sum(P) times in the matrix R. 
%    All values of P should => 0, NaNs are set to 0.
%
%    The other arguments to RANDP specify the size of R in the same way as
%    matlab's own functions do: RANDP(P, N) returns an N-by-N matrix,
%    RANDP(P,M,N) and RANDP(P, [M N]) return M-by-N arrays, etc.
%
%    Examples:
%       % random values from [1 2 4] and a bias for 2
%       R = randp([1 2 0 1], 1, 100) ;  % 100 values
%       histc(R, 1:4)            % -> ~25 ~50 0 ~25
%       
%       % create a random, but biased DNA sequence
%       C ='AGCT', P = [4 1 1 2]
%       DNA = C(randp(P, 1, 50))
%       
%    Also see RAND, RANDPERM
%             RANDPERMBREAK, RANDINTERVAL, RANDSWAP (MatLab File Exchange)
%             RANDSAMPLE (Stats Toolbox)

% Created for Matlab R13+, last tested in 2018a
% version 3.0 (mar 2019)
% (c) Jos van der Geest
% http://www.mathworks.com/matlabcentral/fileexchange/authors/10584
% email: samelinoa@gmail.com
%
% File history:
% 1.0 (nov 2005) - created
% 1.1 (nov 2005) - modified slightly to check input arguments to RAND first
% 1.2 (aug 2006) - fixed bug when called with scalar argument P
% 2.0 (feb 2009) - use HISTC for creating the integers (faster and simplier
%                  than previous algorithm)
% 2.1 (dec 2017) - updated for newer releases
% 2.2 (nov 2018) - updated comments, use cumsum more efficiently
% 3.0 (mar 2019) - updated to avoid cumulative round-off errors in weights,
%                  added check on P, updated code, improved help section

narginchk(2,Inf) ;

P = P(:) ;
if ~isnumeric(P) || isempty(P) 
    error('RANDP:InvalidProbabilityArgument', ...
        'First argument should be a non-empty numerical array.') ;
elseif any(P < 0) 
    error('RANDP:InvalidProbabilitiesNegative', ...
        'All probabilities should be 0 or larger.') ;
elseif any(isinf(P))
   error('RANDP:InvalidProbabilitiesInf', ...
        'Probabilities should be finite.') ;
end

% let rand do all the argument checking for the other arguments
try
    R = rand(varargin{:}) ;    
catch ME
    rethrow(ME) ;
end

P(isnan(P)) = 0 ;
if all(P == 0)
    warning('RANDP:ZeroProbabilities', ...
        'All zero probabilities -> returning zeros') ;
    R(:) = 0 ;
elseif numel(P) == 1
    % a single probability -> all 1
    R(:) = 1 ;
else
    % use histc with numel(P) bins with specific sizes to bin the elements
    P = cumsum(P) ;
    P = [0 ; P] ./ P(end) ;
    P(end) = 1 ; % exact to eliminate cumulative round-off errors
    [~, R] = histc(R, P) ;
end

% Note that RANDP is older than RANDSAMPLE which adopted this technique :-)
