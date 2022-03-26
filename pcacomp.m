
I = im2double(imread('/home/prakhar/Documents/MTech_Subjects/1stSem/CLA/CLA_PG2_Lectures_Assign/watch.bmp'));

f1=figure;
ax1=axes(f1);
imshow(I,'Parent',ax1);
title(ax1,'Original Image');
hold off;

% Plotting compressed image for different value of K
dim_K=5;
[J,error]=pca(I,dim_K);
f2=figure;
ax2=axes(f2);
imshow(J,'Parent',ax2);
title(ax2,'Compressed image, K=5');
fprintf('Error between I and J for K=%d is %e\n',dim_K,error)
hold off;

dim_K=10;
[J,error]=pca(I,dim_K);
f3=figure;
ax3=axes(f3);
imshow(J,'Parent',ax3);
title(ax3,'Compressed image, K=10');
fprintf('Error between I and J for K=%d is %e\n',dim_K,error)
hold off;

dim_K=20;
[J,error]=pca(I,dim_K);
f4=figure;
ax4=axes(f4);
imshow(J,'Parent',ax4);
title(ax4,'Compressed image, K=20');
fprintf('Error between I and J for K=%d is %e\n',dim_K,error)
hold off;


dim_K=64;
[J,error]=pca(I,dim_K);
f5=figure;
ax5=axes(f5);
imshow(J,'Parent',ax5);
title(ax5,'Compressed image, K=64');
fprintf('Error between I and J for K=%d is %e\n',dim_K,error)
hold off;


%figure;
drawnow
% Part 7; running the code till K=64
fprintf("Checking error for all dimensions till 64\n")
err=zeros(1,64);
for dim=1:64
    fprintf('%d\t',dim);
    [J,error]=pca(I,dim);
    err(1,dim)=error;
end
ax6=nexttile;
plot(ax6,err);
title('Frobneious Norm (Original-Compressed) vs Dimension');
fprintf("From the plot, the Error monotonically decreases\n");

function[J,error]=pca(I,dim_K)
% Some Important Matrices documented below
% Block wise Channels: Blocks_R, Blocks_B, Blocks_G
% Col wise Channels: red, blue, green
% Mean of each channel: mean_red, mean_blue, mean_green
% Covariance Matrices: red_cov, blue_cov, green_cov
% Orthonormal Eigenvectores of Cov matrix, desc order: red_V, blue_V, green_V
% Modified Channels: red_new, blue_new, green_new
% Compressed Matrix: J

%Input K for Compressed dimension
limit=dim_K;
% Part 1: Extracting RBG components from the bit map
%I = im2double(imread('/home/prakhar/Documents/MTech_Subjects/1stSem/CLA/CLA_PG2_Lectures_Assign/watch.bmp'));
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

% Part 2: Diving the data sets into multiple contiguous blocks for all 3 components

a=size(R,1);
b=size(R,2);
% Defining block size
sq_size=8;
% Finding # blocks, rows and col wise
x=a/8;
y=b/8;
part1=ones(x);
part1=part1(1,:).*8;
part2=ones(y);
part2=part2(1,:).*8;
% Creating blocks for each three channels
blocks_R=mat2cell(R,part1,part2);
blocks_B=mat2cell(B,part1,part2);
blocks_G=mat2cell(G,part1,part2);
% Reshaping blocks into column vectors
k=1;
%Since all 3 components have same size; initializing the loop with size of
%any one component
for i=1:size(blocks_R,1)
    for j=1:size(blocks_R,2)
        %Reshaping sq blocks into column vectors
        %sq_size*sq_size tells the total sq matrix components 
        red(:,k)=reshape(blocks_R{i,j},1,(sq_size*sq_size));    %Components    
        blue(:,k)=reshape(blocks_B{i,j},1,(sq_size*sq_size));        
        green(:,k)=reshape(blocks_G{i,j},1,(sq_size*sq_size));
        k=k+1;
    end
end

% Part 3: Calculating Mean Vector for all 3 components

mean_red=sum(red,2)./(12288);                         %Mean vector
mean_blue=sum(blue,2)./(12288);
mean_green=sum(green,2)./(12288);


% Part 4: Calculating sample covariance matrix for all 3 components

%Defining e to find means so as to calculate xi-mean
e=ones((sq_size*sq_size),size(red,2));
mean_matrix_red=e.*mean_red;
red_sub_mean=red-mean_matrix_red;
red_cov=red_sub_mean*red_sub_mean';
red_cov=red_cov/(12288);  % Covariance Matrix for Red Channel
%Covariance Matrix
%Since all 3 components have the same size, we can use 'e' again to
%calculate their covariance matrices
mean_matrix_blue=e.*mean_blue;
blue_sub_mean=blue-mean_matrix_blue;
blue_cov=blue_sub_mean*blue_sub_mean';
blue_cov=blue_cov/(12288);  % Covariance Matrix for Blue Channel        
                    
%Calculating covariance matrix for green component
mean_matrix_green=e.*mean_green;
green_sub_mean=green-mean_matrix_green;
green_cov=green_sub_mean*green_sub_mean';
green_cov=green_cov/(12288); % Covariance Matrix for Green Channel                             

%Part 5: Calculating Orthonormal eigen vectors of sample covariance matrix

C_red = red_cov;
[V_red,Lambda_red] = eig(C_red);
Lambda_red = diag(Lambda_red);
[Lambda_red,indices_red] = sort(Lambda_red,'descend');
red_V = V_red(:,indices_red);    % Orthonormal eigen vectors of Red covariance matrix in descending order

C_blue = blue_cov;
[V_blue,Lambda_blue] = eig(C_blue);
Lambda_blue = diag(Lambda_blue);
[Lambda_blue,indices_blue] = sort(Lambda_blue,'descend');
blue_V = V_blue(:,indices_blue);    % Orthonormal eigen vectors of Blue covariance matrix in descending order


C_green = green_cov;
[V_green,Lambda_green] = eig(C_green);
Lambda_green = diag(Lambda_green);
[Lambda_green,indices_green] = sort(Lambda_green,'descend');
green_V = V_green(:,indices_green);    % Orthonormal eigen vectors of Green covariance matrix in descending order



% Part 6: 

% xi-mean for each channel: red_sub_mean, blue_sub_mean, green_sub_mean
%Inner Product <xi-mean,v>

e=ones(size(red,1),size(red,2));

mean_row_red=mean_red.*e;
% For channel datapoints- mean of that channel
red_mean=red-mean_row_red;
red_new=red;

mean_row_blue=mean_blue.*e;
% For channel datapoints- mean of that channel
blue_mean=blue-mean_row_blue;
blue_new=blue;

mean_row_green=mean_green.*e;
% For channel datapoints- mean of that channel
green_mean=green-mean_row_green;
green_new=green;
for i=1:size(red,2)
    sum_vector_green=zeros(size(green_V,1),1);
    sum_vector_blue=zeros(size(blue_V,1),1);
    sum_vector_red=zeros(size(red_V,1),1);    
    for k=1:limit
        % For Red Channel
        inner_prod_red=red_mean(:,i)'*red_V(:,k);
        red_ortho_k=inner_prod_red.*red_V(:,k);
        sum_vector_red=sum_vector_red+red_ortho_k;
        % For Blue Channel
        inner_prod_blue=blue_mean(:,i)'*blue_V(:,k);
        blue_ortho_k=inner_prod_blue.*blue_V(:,k);
        sum_vector_blue=sum_vector_blue+blue_ortho_k;
        % For Green Channel
        inner_prod_green=green_mean(:,i)'*green_V(:,k);
        green_ortho_k=inner_prod_green.*green_V(:,k);
        sum_vector_green=sum_vector_green+green_ortho_k;                
    end
    % Creating New Channels
    red_new(:,i)=sum_vector_red+mean_red;
    blue_new(:,i)=sum_vector_blue+mean_blue;
    green_new(:,i)=sum_vector_green+mean_green;
end



% Creating Blocks all these Channels

% For Red Channel
k=1;
for i=1:96
    for j=1:128
        shape_red=red_new(:,k);
        blocks_R_new{i,j}=reshape(shape_red,8,8);
        k=k+1;
    end
end

% For Blue Channel
k=1;
for i=1:96
    for j=1:128
        shape_blue=blue_new(:,k);
        blocks_B_new{i,j}=reshape(shape_blue,8,8);
        k=k+1;
    end
end

% For Green Channel
k=1;
for i=1:96
    for j=1:128
        shape_green=green_new(:,k);
        blocks_G_new{i,j}=reshape(shape_green,8,8);
        k=k+1;
    end
end

% Combining the blocks to form the matrix
cell_matrix_red=cell2mat(blocks_R_new);
cell_matrix_blue=cell2mat(blocks_B_new);
cell_matrix_green=cell2mat(blocks_G_new);

% Combining all 3 channels together
J=zeros(size(I));
J(:,:,1)=cell_matrix_red;
J(:,:,2)=cell_matrix_green;
J(:,:,3)=cell_matrix_blue;
% Calculating Error between original and compressed image
I_J=I-J;
I_J_2=I_J.^2;
error=sqrt(sum(sum(sum(I_J_2))));
end
