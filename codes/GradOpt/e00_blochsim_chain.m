clear all; close all;

dT = 1;		         % 1ms delta-time.
T = 1000;	         % total duration
N = ceil(T/dT)+1;    % number of time steps.
df = 10;	         % Hz off-resonance.
T1 = 600;            % ms.
T2 = 100;	         % ms.

% RELAXATION operator
R = @(t,T1,T2) [exp(-t/T2) 0            0          0;
                    0      exp(-t/T2)   0          0;
                    0      0            exp(-t/T1) 1-exp(-t/T1);
                    0      0            0          1];

% FLIP operator
F = @(alpha) [cos(alpha*pi/180)  0 sin(alpha*pi/180) 0;
             0                   1 0                 0;
             -sin(alpha*pi/180)  0 cos(alpha*pi/180) 0;
             0                   0 0                 1];
      
% PRECESS operator
P = @(phi,df) [cos((phi+df)*pi/180) -sin((phi+df)*pi/180)  0 0;
              sin((phi+df)*pi/180)  cos((phi+df)*pi/180)   0 0;
              0                     0                      1 0;
              0                     0                      0 1];
          
%2*pi*df*T/1000

M0 = zeros(4,1); M0([1,4],1) = 1;
M = P(10,df)*R(10,T1,T2)*F(10)*M0;














