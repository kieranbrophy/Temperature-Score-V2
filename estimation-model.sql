select distinct assetid, estimated_scope_1_emissions from public.emissions_estimate_scope_1__1_0_0_0
where date_created > '2022-12-05'
order by assetid asc