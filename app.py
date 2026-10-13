import gradio as gr
import pandas as pd
import joblib

# LOAD TRAINED MODEL
MODEL_PATH = "loan_naive_bayes_model.pkl"

model = joblib.load(MODEL_PATH)

# LOAN PREDICTION FUNCTION

def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicantincome,
    coapplicantincome,
    loanamount,
    loan_amount_term,
    credit_history,
    property_area
):

    #BASIC INPUT VALIDATION

    if applicantincome is None:
        return "⚠️ Please enter Applicant Income.", ""

    if coapplicantincome is None:
        coapplicantincome = 0

    if loanamount is None:
        return "⚠️ Please enter Loan Amount.", ""

    if loan_amount_term is None:
        return "⚠️ Please enter Loan Amount Term.", ""

    if credit_history is None:
        return "⚠️ Please select Credit History.", ""


    # Convert numeric values
    applicantincome = float(applicantincome)
    coapplicantincome = float(coapplicantincome)
    loanamount = float(loanamount)
    loan_amount_term = float(loan_amount_term)
    credit_history = int(credit_history)

# NEGATIVE VALUE VALIDATION
    

    if applicantincome < 0:
        return "❌ Applicant income cannot be negative.", ""

    if coapplicantincome < 0:
        return "❌ Coapplicant income cannot be negative.", ""

    if loanamount <= 0:
        return "❌ Loan amount must be greater than 0.", ""

    if loan_amount_term <= 0:
        return "❌ Loan term must be greater than 0.", ""

    # TOTAL INCOME

    total_income = applicantincome + coapplicantincome


    if total_income <= 0:
        return (
            "🔴 LOAN NOT APPROVED",
            "Total applicant income must be greater than 0."
        )

    
    # The original dataset contains relatively small loan
    # amounts compared with extreme values such as
    # 1,000,000,000.
    #
    # We therefore reject clearly unrealistic loan requests
    # instead of allowing the ML model to extrapolate blindly.
    #
    # This is NOT the Naive Bayes model itself.
    # It is an input validation/business-rule layer.

    loan_income_ratio = loanamount / total_income

    if loan_income_ratio > 100:
        return (
            "🔴 LOAN NOT APPROVED",
            f"Requested loan is extremely high compared with income.\n\n"
            f"Total Income: {total_income:,.2f}\n"
            f"Loan Amount: {loanamount:,.2f}\n"
            f"Loan / Income Ratio: {loan_income_ratio:,.2f}x\n\n"
            "⚠️ The requested loan amount is outside a reasonable "
            "range for this ML prediction."
        )

    input_data = pd.DataFrame({
        "gender": [gender],
        "married": [married],
        "dependents": [dependents],
        "education": [education],
        "self_employed": [self_employed],
        "applicantincome": [applicantincome],
        "coapplicantincome": [coapplicantincome],
        "loanamount": [loanamount],
        "loan_amount_term": [loan_amount_term],
        "credit_history": [credit_history],
        "property_area": [property_area]
    })
    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)
    if 1 in classes:
        approved_probability = (
            probabilities[classes.index(1)] * 100
        )
    else:
        approved_probability = 0


    if 0 in classes:
        rejected_probability = (
            probabilities[classes.index(0)] * 100
        )
    else:
        rejected_probability = 0

    if prediction == 1:

        result = "🟢 LOAN APPROVED"

    else:

        result = "🔴 LOAN NOT APPROVED"


    # ========================================================
    # RESULT DETAILS
    # ========================================================

    probability_text = (
        f"Approved Probability : {approved_probability:.2f}%\n"
        f"Rejected Probability : {rejected_probability:.2f}%\n\n"
        f"Total Income         : {total_income:,.2f}\n"
        f"Loan Amount          : {loanamount:,.2f}\n"
        f"Loan / Income Ratio  : {loan_income_ratio:.2f}x"
    )


    return result, probability_text

# GRADIO UI

with gr.Blocks(
    title="Loan Approval Prediction"
) as demo:

    gr.Markdown(
        """
        # 🏦 Loan Approval Prediction

        ### Gaussian Naive Bayes Machine Learning Model

        Enter applicant information to predict whether the
        loan application is likely to be approved or rejected.
        """
    )
    gr.Markdown("## 👤 Applicant Information")


    with gr.Row():

        gender = gr.Dropdown(
            choices=["Male", "Female"],
            label="Gender",
            value="Male"
        )

        married = gr.Dropdown(
            choices=["Yes", "No"],
            label="Married",
            value="Yes"
        )

        dependents = gr.Dropdown(
            choices=["0", "1", "2", "3+"],
            label="Dependents",
            value="0"
        )


    with gr.Row():

        education = gr.Dropdown(
            choices=["Graduate", "Not Graduate"],
            label="Education",
            value="Graduate"
        )

        self_employed = gr.Dropdown(
            choices=["Yes", "No"],
            label="Self Employed",
            value="No"
        )

        property_area = gr.Dropdown(
            choices=["Urban", "Semiurban", "Rural"],
            label="Property Area",
            value="Urban"
        )

    gr.Markdown("## 💰 Financial Information")


    with gr.Row():

        applicantincome = gr.Number(
            label="Applicant Income",
            value=5000,
            minimum=0
        )

        coapplicantincome = gr.Number(
            label="Coapplicant Income",
            value=1500,
            minimum=0
        )

        loanamount = gr.Number(
            label="Loan Amount",
            value=150,
            minimum=1
        )


    with gr.Row():

        loan_amount_term = gr.Number(
            label="Loan Amount Term",
            value=360,
            minimum=1
        )

        credit_history = gr.Dropdown(
            choices=[1, 0],
            label="Credit History",
            value=1
        )
    

    predict_button = gr.Button(
        "🔍 Predict Loan Approval",
        variant="primary"
    )


    gr.Markdown("## 📊 Prediction Result")


    prediction_output = gr.Textbox(
        label="Loan Decision",
        interactive=False
    )


    probability_output = gr.Textbox(
        label="Prediction Details",
        lines=7,
        interactive=False
    )


    predict_button.click(
        fn=predict_loan,

        inputs=[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicantincome,
            coapplicantincome,
            loanamount,
            loan_amount_term,
            credit_history,
            property_area
        ],

        outputs=[
            prediction_output,
            probability_output
        ]
    )


    gr.Markdown(
        """
        ---

        **Model:** Gaussian Naive Bayes  
        **Application:** Loan Approval Prediction  
        **Framework:** Gradio  
        **Deployment:** Local Host
        """
    )
if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )