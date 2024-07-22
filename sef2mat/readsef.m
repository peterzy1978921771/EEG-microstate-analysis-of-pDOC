function [data,hdr] = readsef(filename,flex)
% READSEF  reads Carool's Simple EEG Data Format.
% Usages:
%   [data,hdr] = readsef(filename)
%   [data,hdr] = readsef(filename,samples)
%   hdr  = readsef(filename,'hdr')
%   data = readsef(hdr)
%   data = readsef(hdr,samples)
%
% hdr       structure containing file header data
% data      structure containing data
% samples   Nx2 matrix with start and end samples of the data to be read.
%           Default is entire dataset.
%
% (c) 2008 A. Guggisberg


% Defaults and deal with parameters
doreadhdr=true;
doreaddata=true;
doselectsamples=false;
if isstruct(filename)
    doreadhdr=false; 
    hdr=filename;
    filename=hdr.filename;
end
if nargin>1 && ischar(flex)
   doreaddata=false;
end
if nargin>1 && isnumeric(flex)
    doselectsamples=true;
    samples = flex;
    numtr = size(samples,1);
end
 
% open filename for reading
fid=fopen(filename,'r');

if doreadhdr   
    % read fixed part of header
    hdr.filename    = filename;
    hdr.version     = strcat(fread(fid,4,'int8=>char')');   % 4 bytes
    hdr.numchan     = fread(fid,1,'int32');                 % 4 bytes
    hdr.numauxchan  = fread(fid,1,'int32');                 % 4 
    hdr.numsamples  = fread(fid,1,'int32');                 % 4
    hdr.srate       = fread(fid,1,'float32');               % 4
    hdr.year        = fread(fid,1,'int16');                 % 2
    hdr.month       = fread(fid,1,'int16');                 % 2
    hdr.day         = fread(fid,1,'int16');                 % 2
    hdr.hour        = fread(fid,1,'int16');                 % 2
    hdr.minute      = fread(fid,1,'int16');                 % 2
    hdr.second      = fread(fid,1,'int16');                 % 2
    hdr.millisecond = fread(fid,1,'int16');                 % 2
    
    % read variable part of header
    hdr.sensor_labels = fread(fid,[8 hdr.numchan],'int8=>char')';    % numchannels * 8
elseif ~doselectsamples
    fseek(fid, 34 + 8*hdr.numchan, 'bof');
end

if doreaddata
    % read data
    if doselectsamples
        data.data=cell(numtr,1);
        for k=1:numtr
            fseek(fid, 34 + 8*hdr.numchan + hdr.numchan*(samples(k,1)-1)*4, 'bof');
            currnumsamples = diff(samples(k,:))+1;
            data.data{k} = fread(fid,[hdr.numchan currnumsamples],'float32')';
        end
    else
        data.data{1} = fread(fid,[hdr.numchan hdr.numsamples],'float32')';
    end
    data.srate = hdr.srate;
    %data.latency = 1000 .* [0:1/data.srate:SECPERTRIAL-1/data.fsample]';    
    data.sensor_labels=cell(hdr.numchan,1);
    for k=1:hdr.numchan
        data.sensor_labels{k} = deblank(hdr.sensor_labels(k,:));
    end
    data.filename = hdr.filename;
else
    data=hdr;
end

% close file
fclose(fid);
