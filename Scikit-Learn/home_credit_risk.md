# Goal 

To predict Home credit's clients repayment abilities. The Machine learning training will ensure that clients capable of repayment are not rejected and that loans are given with a repayment calendar that will empower their clients to be succesful. 

# Data 

* application_{train|test}.csv = one row represents one loan in our data sample 

* bureau.csv = all client's previous credit provided by OTHER financial institutions that were reported to Credit BUREAU (one row for every row the client had before the application date). 

* bureau_balance.csv = Monthly balances of previous credit in BAREAU. 
 --- Months_Balance column: relative time index (0 = current month at the time of application, -1 = one month ago, ...)

* POS_CASH_balance.csv = Monthly balance of previous point of sales (POS) and cash loans with HOME CREDIT. 
---  CNT_INSTALMENT = Total number of installments originally agreed for the loan (Ex. 24 = The customer was supposed to pay this loan in 24 installments)
--- CNT_INSTALMENT_FUTURE = How many installments were still left at that month.

* credit_card_balance.csv = Monthly balance snapshots of previous credit cards that applicant has in HOME CREDIT 

* previous_applications.csv = All previous applications for HOME CREDIT loans of clients who have loans in our sample.

* installments_payments.csv = Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample

* HomeCredit_columns_description.csv = This file contains descriptions for the columns in the various data files.

# Relationships

key_column: 

SK_ID_CURR = current_customer ID


Customer Application
     |
     +--- External credit history
     |     |
     |     +--- Monthly bureau behavior 
     |
     +--- Previous HOME CREDIT loans
            |
            +--- POS monthly behavior
            +--- Installment payments
            +--- credit card balances


## how train and other tables are related

`application_train`: what we know about the customer NOW

Historical information

--- `bureau`: external credit history (1-to-Many)
--- `bureau_balance`: montly external loan behavior (1-to-Many)
--- `previous aplications`: previous home credit loans (1-to-Many)
--- `POSH_CASH_balance`: Monthly POS/cash loan behavior (1-to-Many)
--- `installments_payments`: Installment payment history (1-to-Many)
--- `credit_card_balance`: Credit card behavior (1-to-Many)

# TARGET information

`TARGET = 1` means the client had payments difficulties
`TARGET = 0` means the client repaid the loan normally 

# Train Data Column information (most important)

## Finantial Variables
`AMT_INCOME_TOTAL`: Total annual income (numerical)
`AMT_CREDIT`: Loan amount requested (numerical)
`AMT_ANNUITY`: Loan annuity/payment amount (numerical)
`AMT_GOODS_PRICE`: Price of goods financed by the loan (numerical)

## Personal information
`CNT_CHILDREN`: Number of children (numerical)
`DAYS_BIRTH`: Customer age in days (numerical)
`NAME_FAMILY_STATUS` (categorical)
`NAME_EDUCATION_TYPE` (categorical)
`NAME_HOUSING_TYPE` (categorical)
`CNT_FAM_MEMBERS` (numerical)

## Professional Information
`NAME_INCOME_TYPE` (categorical)
`OCCUPATION_TYPE` (categorical)
`ORGANIZATION_TYPE` (categorical)
`DAYS_EMPLOYED`: Number of days employed (numerical)

## Geographic
`REGION_RATING_CLIENT` (numerical)
`REGION_RATING_CLIENT_W_CITY` (numerical)

## External risk scores
`EXT_SOURCE_1` (numerical)
`EXT_SOURCE_2` (numerical)
`EXT_SOURCE_3` (numerical)

## Housing building information

APARTMENTS_AVG
BASEMENTAREA_AVG
YEARS_BUILD_AVG
LIVINGAREA_AVG

## Document flags
`Flag_Document_i`: 0 Document not provided, 1 Document provided

## Credit Bureau Variables
`AMT_REQ_CREDIT_BUREAU_HOUR`: Number of bureau inquiries in last hour

