function [inds] = combine_threshold_crossings(indspos, indsneg, ...
                                              combinationmode, w)
% combine the positive and negative threshold crossings into a single
% array of indices
%
% indspos         - indices of positive threshold crossings
% indsneg         - indices of negative threshold crossings
% combinationmode - string specifying how to combine indspos and
%                   indsneg
% w               - width in samples for closing samples in close
%                   proximity
    
% combine the indices
switch combinationmode
    case 'union'
        inds1 = union(indspos, indsneg);            
    case 'pos'
        inds1 = indspos;
    case 'neg'
        inds1 = indsneg;
    otherwise
        error([combinationmode, ' not implemented in ', mfilename]);
end

% convert the inds to a binary vector
indsvec = zeros(1, max(inds1), 'uint8');
indsvec(inds1) = 1;
% perform a morphological closing to group indices in close proximity
closed = imclose(indsvec, ones(1, w));
% find the connected components from the black and white array
cc = bwconncomp(closed);
% get the first index of each connected component
inds = zeros(1, length(cc.PixelIdxList));
for ii = 1:length(cc.PixelIdxList)
    pixidlist = cc.PixelIdxList{ii};
    inds(ii) = pixidlist(ceil(length(pixidlist)/2));
end
           