function [OA, AA, kappa, ca] = p1_metrics(y_test, y_pred, class_order)
%P1_METRICS Same confusionmat / OA / AA / kappa code as SuperSDF_MC10.m.
cm = confusionmat(y_test, y_pred, 'Order', class_order);
OA = 100 * trace(cm) / sum(cm(:));
ca = 100 * diag(cm) ./ max(sum(cm,2),1);
AA = mean(ca);
N = sum(cm(:)); po = trace(cm)/N; pe = sum(sum(cm,2) .* sum(cm,1)') / N^2;
if abs(1-pe) < eps, kappa = 0; else, kappa = 100*(po-pe)/(1-pe); end
end
